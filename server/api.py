"""FastAPI 入口：REST + SSE 只读端点（Task 5 范围）。

写端点（quantize/eval/...）是 Task 6。所有端点复用 webui.runner /
webui.vocab / server.capabilities / server.jobs 的既有 API，本模块不
重复实现探测、拼命令、任务管理逻辑；量化命令串一律走 webui.cmds.build_cmd
（golden 钉死）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from webui import progress, runner, vocab
from webui.cmds import build_cmd, prettify_cmd, result_tables
from webui.config import load_config

from server.capabilities import probe_capabilities, validate_cmd_flags
from server.jobs import Job, JobRegistry
from server.schemas import PreviewRequest

CFG = load_config()
REGISTRY = JobRegistry(CFG)
REGISTRY.reload()   # 启动重挂：收编上次进程遗留的 job.json

router = APIRouter(prefix="/api")
app = FastAPI(title="ModelOpt Quant WebUI")
app.include_router(router)


# ---------------------------------------------------------------- 辅助

def _paths() -> dict:
    """本机路径四件套（config.yaml local 段原始字符串，不解析绝对路径）。"""
    return {
        "repo": CFG.local.repo,
        "venv": CFG.local.venv,
        "task_dir": CFG.local.task_dir,
        "hf_cache": CFG.local.hf_cache,
    }


def _net() -> dict:
    return {"mode": CFG.net.mode, "proxy": CFG.net.proxy}


def _job_summary(j: Job) -> dict:
    return {
        "id": j.id,
        "kind": j.kind,
        "state": j.state,
        "started_at": j.started_at,
        "export_dir": j.export_dir,
        "version": j.version,
        "commit": j.commit,
        "cmd_preview": j.cmd[:120],
    }


def _scan_vocab() -> dict:
    try:
        qf = vocab.scan_qformats(CFG.local.repo)
    except Exception as e:
        return {"qformats": [], "kv": [list(x) for x in vocab.kv_choices()],
                "datasets": _datasets_list(), "combos": vocab.COMBO_DESC,
                "error": str(e)}
    return {"qformats": qf.get("model", []), "kv": [list(x) for x in vocab.kv_choices()],
            "datasets": _datasets_list(), "combos": vocab.COMBO_DESC,
            "source": qf.get("source", ""), "error": qf.get("error", "")}


def _datasets_list() -> list[dict]:
    """把 vocab.DATASETS 规整成 [{key, desc}]。

    DATASETS 现在是 list[{"key","desc"}]（webui/vocab.py）；兼容 dict
    {key: desc} 形态，避免以后改结构悄悄炸。
    """
    ds = vocab.DATASETS
    if isinstance(ds, dict):
        return [{"key": k, "desc": v} for k, v in ds.items()]
    return [{"key": d.get("key", ""), "desc": d.get("desc", "")} for d in ds]


def _du_size(d: Path) -> str:
    rc, out, _ = runner.run_shell(f"du -sh {runner.q_remote(str(d))}")
    return out.split()[0] if rc == 0 and out.split() else "?"


# ---------------------------------------------------------------- 健康检查

@router.get("/health")
def health() -> dict:
    return {"ok": True}


# ---------------------------------------------------------------- 只读端点

@router.get("/bootstrap")
def bootstrap() -> dict:
    return {
        "paths": _paths(),
        "net": _net(),
        "probe": dataclasses.asdict(runner.probe_host(CFG)),
        "vocab": _scan_vocab(),
        "capabilities": probe_capabilities(CFG),
        "jobs": [_job_summary(j) for j in REGISTRY.list()],
        "defaults": CFG.defaults,
    }


@router.post("/probe")
def probe() -> dict:
    return {
        "probe": dataclasses.asdict(runner.probe_host(CFG)),
        "vocab": _scan_vocab(),
    }


@router.get("/vocab")
def get_vocab() -> dict:
    return _scan_vocab()


@router.get("/capabilities")
def get_capabilities() -> dict:
    return probe_capabilities(CFG)


@router.post("/preview")
def preview(req: PreviewRequest) -> dict:
    p = req.to_build_params_dict()
    try:
        cmd = build_cmd(p, CFG)
    except ValueError as e:
        raise HTTPException(400, detail={"code": "bad_params", "message": str(e), "detail": ""}) from e
    pretty = prettify_cmd(cmd)
    flags_all = probe_capabilities(CFG)["flags_all"]
    missing = validate_cmd_flags(cmd, flags_all)
    if missing:
        raise HTTPException(400, detail={"code": "unknown_flags",
                                         "message": "本版本没有这些参数",
                                         "detail": missing})
    return {"cmd": cmd, "pretty": pretty, "missing_flags": []}


@router.get("/jobs")
def list_jobs() -> dict:
    jobs = REGISTRY.list()   # 已按「活跃在前、各自 started_at 倒序」排好
    active = [j for j in jobs if j.state in ("starting", "running")]
    history = [j for j in jobs if j.state not in ("starting", "running")]
    return {"active": [_job_summary(j) for j in active],
            "history": [_job_summary(j) for j in history]}


@router.get("/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, detail={"code": "no_job",
                                         "message": f"job 不存在: {job_id}",
                                         "detail": ""})
    # quant/eval 有 Task；venv/eval_env 是包装线程（task is None），直接读 log_path
    if job.task is not None:
        log_tail = runner.read_log(job.task, 30000)
        prog = progress.parse(log_tail)
    else:
        log_tail = runner.tail_log(job.log_path, 30000)
        prog = progress.parse_install(log_tail.splitlines())
    result_tables_out = (result_tables(runner.fetch_eval_results(str(CFG.local.task_path)))[0]
                         if job.kind == "eval" else None)
    return {"job": _job_summary(job), "state": job.state, "log_tail": log_tail,
            "progress": dataclasses.asdict(prog),
            "result_tables": result_tables_out}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, detail={"code": "no_job",
                                         "message": f"job 不存在: {job_id}",
                                         "detail": ""})
    q, start = REGISTRY.subscribe(job, request.headers.get("Last-Event-ID", ""))
    return StreamingResponse(REGISTRY.sse_stream(job, q, start),
                             media_type="text/event-stream")


@router.get("/eval/results")
def eval_results() -> dict:
    results = runner.fetch_eval_results(str(CFG.local.task_path))
    rows, note = result_tables(results)   # 单表消费全部（[量化, 基线] 两文件）
    return {"rows": rows, "note": note, "files": [r["file"] for r in results]}


@router.get("/paths")
def get_paths() -> dict:
    return {"paths": _paths()}


@router.get("/net")
def get_net() -> dict:
    return {"net": _net()}


@router.get("/models/local")
def models_local() -> dict:
    """本机 HF 缓存里的模型列表（C1/C2 最小版：目录名 + 大小 + 是否空壳）。"""
    hub = CFG.local.hf_cache_path / "hub"
    if not hub.exists():
        return {"models": [], "error": f"缓存目录不存在: {hub}"}
    out = []
    for d in sorted(hub.glob("models--*--*")):
        if not d.is_dir():
            continue
        out.append({
            "id": d.name.replace("models--", "", 1).replace("--", "/", 1),
            "size": _du_size(d),
            "stub": not (d / "snapshots").exists(),
        })
    return {"models": out, "error": ""}


@router.get("/artifacts")
def artifacts() -> dict:
    """task_dir 下的产物目录（有 run.log 才算一个任务产物）。"""
    base = Path(CFG.local.task_dir).expanduser()
    if not base.exists():
        return {"artifacts": []}
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d / "run.log").exists():
            continue
        state, model, ts = "manual", "", ""
        jj = d / "job.json"
        if jj.exists():
            try:
                meta = json.loads(jj.read_text())
                state = meta.get("state", "manual")
                model = meta.get("export_dir", "")
                ts = meta.get("started_at", "")
            except (OSError, ValueError):
                pass   # job.json 半截/脏数据：当手动产物处理，不炸列表
        out.append({"dir": d.name, "model": model, "ts": str(ts),
                    "size": _du_size(d), "state": state})
    return {"artifacts": out}

"""FastAPI 入口：REST + SSE 全端点（Task 5 只读 + Task 6 写端点）。

所有端点复用 webui.runner / webui.vocab / webui.validate /
server.capabilities / server.jobs 的既有 API，本模块不重复实现探测、
拼命令、任务管理逻辑；量化命令串一律走 webui.cmds.build_cmd（golden 钉死）。
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import threading
import time
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from webui import progress, runner, validate, vocab
from webui.cmds import (build_cmd, build_eval_script, check_local_model,
                        is_local_model, prettify_cmd, result_tables, task_slug)
from webui.config import (DEFAULT_HF_CACHE, DEFAULT_REPO, DEFAULT_TASK_DIR,
                          DEFAULT_VENV, LocalConfig, NetConfig, load_config,
                          save_local, save_net)

from server.capabilities import probe_capabilities, validate_cmd_flags
from server.estimate import estimate as estimate_vram
from server.jobs import Job, JobRegistry
from server.schemas import (BaselineRequest, EvalStart, NetRequest,
                            PathsRequest, PreviewRequest, RecipeDiffRequest,
                            QuantizeParams)
from server.static import mount_static

CFG = load_config()
REGISTRY = JobRegistry(CFG)
REGISTRY.reload()   # 启动重挂：收编上次进程遗留的 job.json

router = APIRouter(prefix="/api")
app = FastAPI(title="ModelOpt Quant WebUI")
app.include_router(router)

# token 中间件：CFG.server_token 非空时所有 /api/* 要求 token。
# header X-Api-Token 优先；缺失时回落 query ?token=（EventSource 不能带自定义
# header，SSE 连接用它）。/api/health 纳管不豁免；token 默认空 ⇒ 本机无感。
# 注意：query 里的 token 会进访问日志，文档已注明。
@app.middleware("http")
async def _token_mw(request, call_next):
    if CFG.server_token and request.url.path.startswith("/api"):
        provided = request.headers.get("X-Api-Token") \
            or request.query_params.get("token")
        if provided != CFG.server_token:
            return JSONResponse(status_code=401, content={"error": {
                "code": "bad_token", "message": "X-Api-Token / ?token= 不符", "detail": ""}})
    return await call_next(request)

mount_static(app)


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
    caps = probe_capabilities(CFG)
    # caps 失败（repo 不可读等）时 flags_all=[] 会把合法命令误判成
    # unknown_flags：error 非空则跳过差集校验，把错误透出给前端。
    missing = [] if caps.get("error") else validate_cmd_flags(cmd, caps["flags_all"])
    if missing:
        raise HTTPException(400, detail={"code": "unknown_flags",
                                         "message": "本版本没有这些参数",
                                         "detail": missing})
    return {"cmd": cmd, "pretty": pretty, "missing_flags": [],
            "caps_error": caps.get("error", "")}


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


@router.get("/estimate")
def estimate_api(
    model: str = "",
    qformat: str = "",
    kv_qformat: str = "",
    ctx: int = 2048,
    batch: int = 1,
) -> dict:
    """显存估算（设计稿 §6）：权重量（缓存 du / HF API）+ 量化折算 + KV（config.json / 经验式）。

    model 空 → 400 bad_params；其余参数用默认值，估算结果带 source 标注。
    """
    if not model.strip():
        raise HTTPException(400, detail={"code": "bad_params", "message": "model 不能为空"})
    return estimate_vram(model.strip(), qformat.strip(), kv_qformat.strip(), ctx, batch, CFG)


@router.get("/models/local")
def models_local() -> dict:
    """本机 HF 缓存里的模型列表（C2 完整版：目录名 + 大小 + 空壳标记 + snapshot revision）。"""
    hub = CFG.local.hf_cache_path / "hub"
    if not hub.exists():
        return {"models": [], "error": f"缓存目录不存在: {hub}"}
    out = []
    for d in sorted(hub.glob("models--*--*")):
        if not d.is_dir():
            continue
        # refs/main 里是一行 snapshot hash（取不到 = 空壳/不完整，留空串）
        revision = ""
        ref = d / "refs" / "main"
        if ref.exists():
            try:
                lines = ref.read_text().strip().splitlines()
                if lines:
                    revision = lines[0]
            except OSError:
                pass
        out.append({
            "id": d.name.replace("models--", "", 1).replace("--", "/", 1),
            "size": _du_size(d),
            "stub": not (d / "snapshots").exists(),
            "revision": revision,
        })
    return {"models": out, "error": ""}


@router.get("/artifacts")
def artifacts() -> dict:
    """task_dir 下的产物目录（有 run.log 才算一个任务产物）。

    qformat / model 名从 job.json.cmd（manual 产物从 run.log 首行）正则提取；
    `model` 字段存模型名（--pyt_ckpt_path），取不到留空，`dir` 是目录名。
    """
    base = Path(CFG.local.task_dir).expanduser()
    if not base.exists():
        return {"artifacts": []}
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d / "run.log").exists():
            continue
        state, model, ts = "manual", "", ""
        qformat = ""
        cmd = ""
        jj = d / "job.json"
        if jj.exists():
            try:
                meta = json.loads(jj.read_text())
                state = meta.get("state", "manual")
                ts = meta.get("started_at", "")
                cmd = meta.get("cmd", "")
            except (OSError, ValueError):
                pass   # job.json 半截/脏数据：当手动产物处理，不炸列表
        if not cmd:   # manual 产物：run.log 首行是完整命令
            try:
                with open(d / "run.log", "rb") as f:
                    cmd = f.readline().decode(errors="replace")
            except OSError:
                cmd = ""
        qformat = _first_match(r"--qformat (\S+)", cmd)
        model = _first_match(r"--pyt_ckpt_path (\S+)", cmd)
        out.append({"dir": d.name, "model": model, "qformat": qformat,
                    "ts": str(ts), "size": _du_size(d), "state": state})
    return {"artifacts": out}


def _first_match(pattern: str, text: str) -> str:
    """首个正则匹配组，取不到返回空串。"""
    m = re.search(pattern, text or "")
    return m.group(1) if m else ""


# ---------------------------------------------------------------- 写端点（Task 6）

def _caps() -> dict:
    """capabilities 探针（失败不炸，flags_all 空时调用方应跳过校验）。"""
    return probe_capabilities(CFG)


@router.post("/quantize")
def quantize(req: QuantizeParams) -> JSONResponse:
    """起量化任务：前置校验 → start_task → create/persist/start_poller → 202。"""
    p = req.to_build_params_dict()
    try:
        cmd = build_cmd(p, CFG)
    except ValueError as e:
        raise HTTPException(400, detail={"code": "bad_params",
                                         "message": str(e), "detail": ""}) from e
    probe = runner.probe_host(CFG)
    if not probe.reachable:
        raise HTTPException(400, detail={"code": "unreachable",
                                         "message": probe.error or "目标不可达", "detail": ""})
    if not probe.venv_ok:
        raise HTTPException(400, detail={"code": "venv_missing",
                                         "message": "量化 venv 不存在，先 POST /api/venv",
                                         "detail": ""})
    model = str(p.get("model") or "").strip()
    if is_local_model(model):
        why = check_local_model(CFG, model)
        if why:
            raise HTTPException(400, detail={"code": "bad_model_path",
                                             "message": why, "detail": ""})
    caps = _caps()
    if not caps.get("error"):
        missing = validate_cmd_flags(cmd, caps["flags_all"])
        if missing:
            raise HTTPException(400, detail={"code": "unknown_flags",
                                             "message": "本版本没有这些参数",
                                             "detail": missing})
    try:
        task = runner.start_task(CFG, str(CFG.local.task_path),
                                 task_slug(model), str(p.get("export_path") or ""),
                                 cmd, runner.proxy_for(CFG), lambda s: None)
    except Exception as e:
        raise HTTPException(500, detail={"code": "start_failed",
                                         "message": f"启动失败: {e}", "detail": ""}) from e
    job = REGISTRY.create("quant", task, cmd, export_dir=str(p.get("export_path") or ""),
                          version=caps.get("version", ""), commit=caps.get("commit", ""))
    job.state = "running"
    REGISTRY.persist(job)
    REGISTRY.start_poller(job)
    return JSONResponse(status_code=202, content={
        "job_id": job.id, "job": _job_summary(job), "warn_gpu": probe.gpu_busy})


@router.post("/eval")
def eval_start(req: EvalStart) -> JSONResponse:
    """起评测任务：probe_eval_env → build_eval_script → start_script_task → 202。"""
    ok, why = runner.probe_eval_env(CFG)
    if not ok:
        raise HTTPException(400, detail={"code": "eval_env_missing",
                                         "message": why or "评测环境未就绪", "detail": why})
    try:
        script = build_eval_script(req.quant, req.baseline, req.tasks,
                                   max(req.tp, 1), req.port,
                                   str(req.batch or "auto"), CFG)
    except ValueError as e:
        raise HTTPException(400, detail={"code": "bad_params",
                                         "message": str(e), "detail": ""}) from e
    caps = _caps()
    tdir = str(CFG.local.task_path / f"eval_{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        task = runner.start_script_task(CFG, tdir, script, runner.proxy_for(CFG),
                                        lambda s: None, pkill_pattern="vllm serve")
    except Exception as e:
        raise HTTPException(500, detail={"code": "start_failed",
                                         "message": f"启动失败: {e}", "detail": ""}) from e
    job = REGISTRY.create("eval", task, "bash run.sh",
                          version=caps.get("version", ""), commit=caps.get("commit", ""))
    job.state = "running"
    REGISTRY.persist(job)
    REGISTRY.start_poller(job)
    return JSONResponse(status_code=202, content={"job_id": job.id, "job": _job_summary(job)})


def _ensure_job(kind: str, runner_fn, label: str) -> JSONResponse:
    """venv/eval-env 共用的包装线程模式：task=None，poller 靠 _done/_exc 判终态。"""
    job = REGISTRY.create(kind, None, label)
    job.state = "running"

    def _run():
        try:
            runner_fn(lambda s: None, CFG)
        except BaseException as e:  # noqa: BLE001 包装线程必须吞异常交给 poller
            job._exc = e
        finally:
            job._done = True

    threading.Thread(target=_run, daemon=True, name=f"job-{job.id}-ensure").start()
    REGISTRY.persist(job)
    REGISTRY.start_poller(job)
    return JSONResponse(status_code=202, content={"job_id": job.id,
                                                  "job": _job_summary(job)})


@router.post("/venv")
def venv_install() -> JSONResponse:
    """创建量化 venv（ensure_venv 起线程，内部自己 mkstemp 写日志）。"""
    return _ensure_job("venv", runner.ensure_venv, "ensure_venv")


@router.post("/eval-env")
def eval_env_install() -> JSONResponse:
    """装评测环境（vllm + lm_eval）。"""
    return _ensure_job("eval_env", runner.ensure_eval_env, "ensure_eval_env")


@router.post("/jobs/{job_id}/abort")
def abort_job(job_id: str) -> dict:
    """中止 job：REGISTRY.abort（有 task 走 abort_task，无 task 置 _done）。"""
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, detail={"code": "no_job",
                                         "message": f"job 不存在: {job_id}", "detail": ""})
    state = REGISTRY.abort(job_id)
    return {"message": "已发送中止信号", "state": state}


@router.post("/recipe/diff")
def recipe_diff(req: RecipeDiffRequest) -> dict:
    """对比本次量化产物与官方 recipe 结果（对照 legacy do_validate）。"""
    job = REGISTRY.get(req.job_id)
    if job is None:
        raise HTTPException(404, detail={"code": "no_job",
                                         "message": f"job 不存在: {req.job_id}", "detail": ""})
    if job.task is None:
        raise HTTPException(400, detail={"code": "bad_job",
                                         "message": "该 job 没有可对照的日志", "detail": ""})
    log_tail = runner.read_log(job.task, 5000)
    hint = ""
    with contextlib.suppress(Exception):
        hint = job.task.remote_cmd.split("--export_path")[-1].split()[0].strip("'\"")
    mine = validate.scan_export_dir(log_tail, export_dir_hint=hint)
    diff = ""
    if req.official.strip():
        try:
            off = validate.fetch_official_summary(req.official, proxy=runner.proxy_for(CFG))
            diff = "\n\n".join(validate.diff_summaries(mine, off))
        except Exception as e:  # noqa: BLE001 官方对照失败不炸主产物
            diff = f"⚠ 官方对照失败: {e}"
    return {"summary": dataclasses.asdict(mine), "groups": mine.groups,
            "diff": diff, "files": mine.files}


@router.post("/eval/baseline")
def eval_baseline(req: BaselineRequest) -> dict:
    """从量化 checkpoint 的 config.json 抽基线模型名（对照 legacy do_fetch_baseline）。"""
    q = (req.quant_path or "").strip()
    if not q:
        raise HTTPException(400, detail={"code": "bad_quant_path",
                                         "message": "请填量化模型路径", "detail": ""})
    r, out, _ = runner.run_shell(f"cat {runner.q_remote(q)}/config.json 2>/dev/null",
                                 timeout=20)
    if r != 0 or not out.strip():
        raise HTTPException(400, detail={"code": "bad_quant_path",
                                         "message": "读不到量化 checkpoint 的 config.json（检查路径）",
                                         "detail": ""})
    try:
        d = json.loads(out)
        b = (d.get("quantization_config", {}) or {}).get("_name_or_path", "") \
            or d.get("_name_or_path", "")
    except Exception:  # noqa: BLE001 解析失败
        raise HTTPException(400, detail={"code": "bad_quant_path",
                                         "message": "config.json 解析失败", "detail": ""}) from None
    if not b or b == q:
        return {"baseline": "", "message": "config.json 里没有基线模型名，请手填"}
    return {"baseline": b, "message": "OK"}


@router.put("/paths")
def put_paths(req: PathsRequest) -> dict:
    """改本机路径四件套并写回 config.yaml（空字段回填默认值，同 legacy do_save_env）。"""
    try:
        CFG.local = LocalConfig(
            repo=(req.repo or "").strip() or DEFAULT_REPO,
            venv=(req.venv or "").strip() or DEFAULT_VENV,
            task_dir=(req.task_dir or "").strip() or DEFAULT_TASK_DIR,
            hf_cache=(req.hf_cache or "").strip() or DEFAULT_HF_CACHE)
        save_local(CFG.local)
    except Exception as e:
        raise HTTPException(500, detail={"code": "save_failed",
                                         "message": f"保存失败: {e}", "detail": ""}) from e
    return {"message": "已写入 config.yaml（立即生效，无需重启）"}


@router.put("/net")
def put_net(req: NetRequest) -> dict:
    """改网络模式并写回 net.json。"""
    try:
        CFG.net = NetConfig(mode=(req.mode or "").strip() or "auto",
                            proxy=(req.proxy or "").strip())
        save_net(CFG.net)
    except Exception as e:
        raise HTTPException(500, detail={"code": "save_failed",
                                         "message": f"保存失败: {e}", "detail": ""}) from e
    return {"message": "已写入 net.json（立即生效）"}


@router.post("/net/test")
def net_test() -> dict:
    """测当前代理配置是否可用。"""
    try:
        ok, detail = runner.test_proxy(CFG)
    except Exception as e:
        raise HTTPException(500, detail={"code": "test_failed",
                                         "message": f"测试失败: {type(e).__name__}: {e}",
                                         "detail": ""}) from e
    return {"ok": ok, "detail": detail}

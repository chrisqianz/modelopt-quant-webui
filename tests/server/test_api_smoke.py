"""server/api.py 端点冒烟测试（TestClient + monkeypatch，不真跑量化）。

本机无 CUDA：所有 webui.runner 探测/日志/结果函数与 server.capabilities
的探测都 patch 掉；jobs 单例直接调 REGISTRY.create 造数据。
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

import server.api as api
from server.jobs import JobRegistry
from webui import runner as runner_mod
from webui import vocab as vocab_mod
from webui.config import (
    DEFAULT_HF_CACHE,
    DEFAULT_REPO,
    DEFAULT_TASK_DIR,
    DEFAULT_VENV,
    AppConfig,
    LocalConfig,
    NetConfig,
)
from webui.runner import HostStatus


def _make_cfg(tmp_path) -> AppConfig:
    return AppConfig(
        server_host="127.0.0.1", server_port=7861, server_token="",
        local=LocalConfig(repo=str(tmp_path / "repo"), venv=str(tmp_path / "venv"),
                          task_dir=str(tmp_path / "tasks"), hf_cache=str(tmp_path / "hf")),
        net=NetConfig(mode="auto", proxy="http://127.0.0.1:7890"),
        defaults={"calib_size": 1024, "qformat": "nvfp4"},
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient + 全套 monkeypatch（CFG 指向 tmp_path，探测全部造假）。"""
    monkeypatch.setattr(api, "CFG", _make_cfg(tmp_path))
    monkeypatch.setattr(api, "REGISTRY", JobRegistry(api.CFG))

    monkeypatch.setattr(
        api.runner, "probe_host",
        lambda cfg: HostStatus(reachable=True, gpu="RTX 4090", gpu_util=12,
                               gpu_mem="3/24576 MiB", gpu_busy=False, repo_ok=True,
                               venv_ok=False, uv_ok=True, disk_free="1.0T",
                               hf_cache="2.3G", hf_direct_ok=True, hf_proxy_ok=False,
                               proxy_note="直连", error=""))
    monkeypatch.setattr(api.runner, "read_log", lambda task, tail_bytes=20000: "loading checkpoint shards\nQuantized model exported to: /tmp/out\n")
    monkeypatch.setattr(api.runner, "tail_log", lambda path, tail_bytes=20000: "uv pip install ok\n")
    monkeypatch.setattr(api.runner, "fetch_eval_results", lambda outdir: [])
    monkeypatch.setattr(api.runner, "run_shell", lambda cmd, timeout=60: (0, "1.2G\t/path", ""))
    monkeypatch.setattr(
        api.vocab, "scan_qformats",
        lambda repo: {"model": ["nvfp4"], "kv": ["kvfp4"], "source": "snapshot", "error": ""})
    monkeypatch.setattr(
        api, "probe_capabilities",
        lambda cfg: {"version": "v1.0", "commit": "abc1234",
                    "flags_all": ["--qformat", "--pyt_ckpt_path", "--calib_seq",
                                   "--batch_size", "--device", "--dataset",
                                   "--calib_size", "--kv_cache_qformat", "--recipe",
                                   "--attn_implementation", "--export_path",
                                   "--use_seq_device_map"],
                    "mapped": ["--qformat"], "unmapped": [], "missing": [], "error": ""})
    return TestClient(api.app)


# ---------------------------------------------------------------- 端点形状

def test_bootstrap_shape(client):
    r = client.get("/api/bootstrap")
    assert r.status_code == 200
    body = r.json()
    for k in ("paths", "net", "probe", "vocab", "recipes", "capabilities", "jobs", "defaults"):
        assert k in body
    assert set(body["paths"]) == {"repo", "venv", "task_dir", "hf_cache"}
    assert body["net"]["mode"] == "auto"
    assert body["probe"]["reachable"] is True
    assert body["defaults"]["qformat"] == "nvfp4"


def test_probe(client):
    r = client.post("/api/probe")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"probe", "vocab"}
    assert body["probe"]["repo_ok"] is True


def test_vocab(client):
    r = client.get("/api/vocab")
    assert r.status_code == 200
    body = r.json()
    assert body["qformats"] == ["nvfp4"]
    assert body["kv"] == [list(x) for x in vocab_mod.kv_choices()]
    assert all(set(d) == {"key", "desc", "gated"} for d in body["datasets"])
    assert body["combos"] == vocab_mod.COMBO_DESC
    assert body["source"] == "snapshot"
    assert body["error"] == ""


def test_capabilities(client):
    r = client.get("/api/capabilities")
    assert r.status_code == 200
    for k in ("version", "flags_all", "mapped", "unmapped", "missing"):
        assert k in r.json()


def test_preview_ok(client):
    r = client.post("/api/preview", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4",
        "datasets": ["cnn_dailymail"], "dataset_sizes": {"cnn_dailymail": 512},
        "calib_seq": 512, "batch_size": 0,
    })
    assert r.status_code == 200
    body = r.json()
    assert "--qformat" in body["cmd"]
    assert body["missing_flags"] == []
    assert body["caps_error"] == ""
    assert "\n" in body["pretty"]   # 多行展示形态


def test_preview_caps_error_skips_flag_check(client, monkeypatch):
    """Task 5 reviewer 建议 1：caps 失败（repo 不可读）时 flags_all=[] 会把
    合法命令误判 unknown_flags —— error 非空应跳过差集校验。"""
    monkeypatch.setattr(
        api, "probe_capabilities",
        lambda cfg: {"version": "", "commit": "", "flags_all": [],
                     "mapped": [], "unmapped": [], "missing": [],
                     "error": "仓库预设目录不可读"})
    r = client.post("/api/preview", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4",
        "datasets": ["cnn_dailymail"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["missing_flags"] == []
    assert body["caps_error"] == "仓库预设目录不可读"


def test_preview_unknown_flags(client, monkeypatch):
    monkeypatch.setattr(api, "validate_cmd_flags", lambda cmd, flags_all: ["--x"])
    r = client.post("/api/preview", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4",
        "datasets": [], "dataset_sizes": {},
    })
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unknown_flags"
    assert r.json()["detail"]["detail"] == ["--x"]


def test_preview_empty_model(client):
    r = client.post("/api/preview", json={
        "model": "", "qformat": "nvfp4",
        "datasets": [], "dataset_sizes": {},
    })
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_params"


def test_jobs_list(client, tmp_path):
    j1 = api.REGISTRY.create("quant", None, "cmd-a", export_dir=str(tmp_path / "a"))
    j1.state = "running"
    j2 = api.REGISTRY.create("eval", None, "cmd-b")
    j2.state = "done"
    r = client.get("/api/jobs")
    assert r.status_code == 200
    body = r.json()
    assert [j["id"] for j in body["active"]] == [j1.id]
    assert [j["id"] for j in body["history"]] == [j2.id]
    assert body["active"][0]["cmd_preview"].startswith("cmd-a")


def test_job_detail_quant(client, tmp_path):
    log = tmp_path / "run.log"
    log.write_text("loading checkpoint shards\ndone\n")
    task = runner_mod.Task(task_id="t", remote_cmd="cmd", log_path=str(log),
                           pid="1", state="running")
    j = api.REGISTRY.create("quant", task, "cmd-quant")
    r = client.get(f"/api/jobs/{j.id}")
    assert r.status_code == 200
    body = r.json()
    for k in ("job", "state", "log_tail", "progress", "result_tables"):
        assert k in body
    assert body["state"] == "starting"  # 未起 poller，停在初始态
    assert body["result_tables"] is None
    assert "loading checkpoint" in body["log_tail"]
    assert isinstance(body["progress"]["stage"], str)


def test_job_detail_venv(client, tmp_path):
    j = api.REGISTRY.create("venv", None, "uv venv")
    j.log_path = str(tmp_path / "x.log")
    j.state = "done"
    r = client.get(f"/api/jobs/{j.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "done"
    assert body["result_tables"] is None


def test_job_detail_404(client):
    r = client.get("/api/jobs/nope_123")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "no_job"


def test_job_events_404(client):
    r = client.get("/api/jobs/nope_123/events")
    assert r.status_code == 404


def test_job_events_sse(client, monkeypatch, tmp_path):
    """起 poller（poll_task 置 done）⇒ 终态广播 state ⇒ sse_stream break，
    TestClient 消费完整个流自然终止；收齐 log 重放 + progress + state 事件。
    假 read_log 阻塞到订阅者就位：poller 首轮立即广播，若 state 广播在
    subscribe 之前发生，SSE 流将永等（竞态挂死）。"""
    import time
    log = tmp_path / "sse.log"
    log.write_text("loading\n")
    task = runner_mod.Task(task_id="t", remote_cmd="cmd", log_path=str(log),
                           pid="1", state="running")
    j = api.REGISTRY.create("quant", task, "cmd-sse")
    monkeypatch.setattr(api.runner, "poll_task",
                        lambda t: setattr(t, "state", "done"))

    def fake_read_log(t, tail_bytes=20000):
        while not j._subscribers:      # 等 SSE 端点 subscribe 完成再让 poller 走
            time.sleep(0.01)
        return "loading\n"

    monkeypatch.setattr(api.runner, "read_log", fake_read_log)
    api.REGISTRY.start_poller(j)
    r = client.get(f"/api/jobs/{j.id}/events")
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    for line in ("event: log", "event: progress", "event: state",
                 'data: {"state": "done"}'):
        assert line in r.text
    for _ in range(50):
        if len(j._subscribers) == 0 and j.state == "done":
            break
        time.sleep(0.1)
    assert j.state == "done"
    assert len(j._subscribers) == 0


def test_eval_results(client):
    r = client.get("/api/eval/results")
    assert r.status_code == 200
    body = r.json()
    assert body == {"rows": [], "note": "（无结果文件）", "files": []}


def test_eval_results_with_data(client, monkeypatch):
    data = {"results": {"gpqa_diamond_cot_zeroshot": {
        "gpqa_diamond_cot_zeroshot,acc,none": 0.62}}}
    monkeypatch.setattr(api.runner, "fetch_eval_results",
                        lambda outdir: [{"file": "results_a.json", "data": data}])
    r = client.get("/api/eval/results")
    assert r.status_code == 200
    body = r.json()
    assert body["files"] == ["results_a.json"]
    assert body["rows"][0][1] == "gpqa_diamond_cot_zeroshot"
    assert body["rows"][0][3] == "0.62"


def test_paths(client, tmp_path):
    r = client.get("/api/paths")
    assert r.status_code == 200
    assert r.json()["paths"] == {
        "repo": str(tmp_path / "repo"), "venv": str(tmp_path / "venv"),
        "task_dir": str(tmp_path / "tasks"), "hf_cache": str(tmp_path / "hf"),
    }


def test_net(client):
    r = client.get("/api/net")
    assert r.status_code == 200
    assert r.json()["net"] == {"mode": "auto", "proxy": "http://127.0.0.1:7890",
                               "hf_token_set": False}


# ---------------------------------------------------------------- 本地模型 / 产物

def test_models_local(client, tmp_path):
    hub = tmp_path / "hf" / "hub"
    (hub / "models--org--name" / "snapshots" / "s1").mkdir(parents=True)
    (hub / "models--org--name" / "refs").mkdir()
    (hub / "models--org--name" / "refs" / "main").write_text("abc123def\n")
    (hub / "models--org--stub").mkdir(parents=True)
    r = client.get("/api/models/local")
    assert r.status_code == 200
    body = r.json()
    assert body["error"] == ""
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["org/name"]["stub"] is False
    assert by_id["org/name"]["size"] == "1.2G"
    assert by_id["org/name"]["revision"] == "abc123def"
    assert by_id["org/stub"]["stub"] is True
    assert by_id["org/stub"]["revision"] == ""


def test_models_local_missing_cache(client, tmp_path):
    r = client.get("/api/models/local")
    assert r.status_code == 200
    assert r.json()["models"] == []
    assert "缓存目录不存在" in r.json()["error"]


def test_artifacts(client, tmp_path):
    tasks = tmp_path / "tasks"
    (tasks / "quant_x_20240101" / "snapshots").mkdir(parents=True)
    (tasks / "quant_x_20240101" / "run.log").write_text("log\n")
    r = client.get("/api/artifacts")
    assert r.status_code == 200
    out = r.json()["artifacts"]
    assert len(out) == 1
    assert out[0]["dir"] == "quant_x_20240101"
    assert out[0]["state"] == "manual"
    assert out[0]["model"] == ""
    assert out[0]["qformat"] == ""
    assert out[0]["size"] == "1.2G"


def test_artifacts_with_job_json(client, tmp_path):
    """job.json.cmd 里能提取模型名与 qformat（C1 完整版）。"""
    tasks = tmp_path / "tasks"
    d = tasks / "eval_y_20240202"
    d.mkdir(parents=True)
    (d / "run.log").write_text("log\n")
    cmd = ("cd /repo && /venv/bin/python examples/hf_ptq/hf_ptq.py "
           "--pyt_ckpt_path org/Qwen3-8B --qformat nvfp4 --dataset cnn_dailymail")
    (d / "job.json").write_text(json.dumps({"state": "done", "cmd": cmd,
                                            "started_at": 1700000000}))
    r = client.get("/api/artifacts")
    out = r.json()["artifacts"]
    assert len(out) == 1
    assert out[0]["state"] == "done"
    assert out[0]["model"] == "org/Qwen3-8B"
    assert out[0]["qformat"] == "nvfp4"
    assert out[0]["ts"] == "1700000000"


def test_artifacts_manual_runlog_cmd(client, tmp_path):
    """manual 产物（无 job.json）：从 run.log 首行提取模型名/qformat。"""
    tasks = tmp_path / "tasks"
    d = tasks / "manual_m_20240303"
    d.mkdir(parents=True)
    (d / "run.log").write_text(
        "/venv/bin/python examples/hf_ptq/hf_ptq.py --pyt_ckpt_path ~/models/Llama-3.1-8B "
        "--qformat int4_awq --dataset cnn_dailymail\nlog line2\n")
    r = client.get("/api/artifacts")
    out = r.json()["artifacts"]
    assert len(out) == 1
    assert out[0]["state"] == "manual"
    assert out[0]["model"] == "~/models/Llama-3.1-8B"
    assert out[0]["qformat"] == "int4_awq"


def test_artifacts_empty_dir(client):
    r = client.get("/api/artifacts")
    assert r.status_code == 200
    assert r.json() == {"artifacts": []}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ---------------------------------------------------------------- 写端点（Task 6）

def _probe(venv_ok=True, reachable=True, gpu_busy=False, error=""):
    return HostStatus(reachable=reachable, gpu="RTX 4090", gpu_util=80 if gpu_busy else 5,
                      gpu_mem="3/24576 MiB", gpu_busy=gpu_busy, repo_ok=True,
                      venv_ok=venv_ok, uv_ok=True, disk_free="1.0T",
                      hf_cache="2.3G", hf_direct_ok=True, hf_proxy_ok=False,
                      proxy_note="", error=error)


def test_quantize_venv_missing(client, monkeypatch):
    monkeypatch.setattr(api.runner, "probe_host", lambda cfg: _probe(venv_ok=False))
    r = client.post("/api/quantize", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4",
        "datasets": ["cnn_dailymail"],
    })
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "venv_missing"


def test_quantize_unreachable(client, monkeypatch):
    monkeypatch.setattr(api.runner, "probe_host", lambda cfg: _probe(reachable=False, error="ssh 不通"))
    r = client.post("/api/quantize", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4", "datasets": [],
    })
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unreachable"


def test_quantize_bad_params_empty_model(client):
    r = client.post("/api/quantize", json={"model": "", "qformat": "nvfp4", "datasets": []})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_params"


def test_quantize_ok_warn_gpu(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api.runner, "probe_host", lambda cfg: _probe(gpu_busy=True))
    monkeypatch.setattr(api.runner, "start_task",
                        lambda *a, **k: runner_mod.Task(task_id="t", remote_cmd="c",
                                                        log_path=str(tmp_path / "x.log"),
                                                        pid="1", state="running"))
    r = client.post("/api/quantize", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4",
        "datasets": ["cnn_dailymail"],
    })
    assert r.status_code == 202
    body = r.json()
    assert body["warn_gpu"] is True
    assert body["job"]["kind"] == "quant"
    assert any(j.kind == "quant" for j in api.REGISTRY.list())


def test_quantize_start_failed(client, monkeypatch):
    monkeypatch.setattr(api.runner, "probe_host", lambda cfg: _probe())
    monkeypatch.setattr(api.runner, "start_task", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = client.post("/api/quantize", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4", "datasets": [],
    })
    assert r.status_code == 500


def test_recipes_endpoint_and_launch_guard(client, tmp_path, monkeypatch):
    """recipe 菜单端点 + 发射前拦截：不存在的 recipe → 400，不产生任务。"""
    root = tmp_path / "repo" / "modelopt_recipes" / "general" / "ptq"
    root.mkdir(parents=True, exist_ok=True)
    (root / "nvfp4_x.yaml").write_text(
        "metadata:\n  description: d\nquantize:\n  algorithm: {method: max}\n")
    r = client.get("/api/recipes")
    assert r.status_code == 200
    assert [x["name"] for x in r.json()["recipes"]] == ["general/ptq/nvfp4_x"]
    b = client.get("/api/bootstrap").json()
    assert b["recipes"]["recipes"][0]["group"] == "官方通用"

    started = []
    monkeypatch.setattr(api.runner, "probe_host", lambda cfg: _probe())
    monkeypatch.setattr(api.runner, "start_task",
                        lambda *a, **k: (started.append(1),
                                         runner_mod.Task(task_id="t", remote_cmd="c",
                                                         log_path=str(tmp_path / "x.log"),
                                                         pid="1", state="running"))[1])
    r = client.post("/api/quantize", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4", "datasets": [],
        "recipe": "general/ptq/nope"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_recipe"
    assert not started and not any(j.kind == "quant" for j in api.REGISTRY.list())
    r = client.post("/api/quantize", json={
        "model": "hf-org/Tiny-124M", "qformat": "nvfp4", "datasets": [],
        "recipe": "general/ptq/nvfp4_x"})
    assert r.status_code == 202 and started


def test_eval_env_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api.runner, "probe_eval_env", lambda cfg: (False, "缺 vllm"))
    r = client.post("/api/eval", json={"quant": str(tmp_path / "q"), "tasks": ["gsm8k"]})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "eval_env_missing"


def test_eval_ok(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api.runner, "probe_eval_env", lambda cfg: (True, ""))
    monkeypatch.setattr(api.runner, "start_script_task",
                        lambda *a, **k: runner_mod.Task(task_id="t", remote_cmd="bash run.sh",
                                                        log_path=str(tmp_path / "e.log"),
                                                        pid="1", state="running"))
    r = client.post("/api/eval", json={"quant": str(tmp_path / "q"), "tasks": ["gsm8k"]})
    assert r.status_code == 202
    assert r.json()["job"]["kind"] == "eval"


def test_venv_install(client, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(api.runner, "ensure_venv",
                        lambda log, cfg: (seen.update(ran=True) or None))
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "_sleep", lambda s: time.sleep(0.01))
    r = client.post("/api/venv")
    assert r.status_code == 202
    jid = r.json()["job_id"]
    j = api.REGISTRY.get(jid)
    assert j is not None
    assert j.kind == "venv" and j.task is None
    # 端到端：ensure_venv 早退不写 /tmp 日志 ⇒ poller 靠 _done=True 判 done
    for _ in range(200):
        if j.state in ("done", "failed"):
            break
        time.sleep(0.05)
    assert seen.get("ran") is True
    assert j.state == "done"


def test_venv_install_with_log(client, monkeypatch, tmp_path):
    """有日志路径（真实安装流）：marker 判定优先于 _done。"""
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "_sleep", lambda s: time.sleep(0.01))
    monkeypatch.setattr(api.runner, "ensure_venv", lambda log, cfg: None)
    r = client.post("/api/venv")
    assert r.status_code == 202
    j = api.REGISTRY.get(r.json()["job_id"])
    assert j is not None
    j.log_path = str(tmp_path / "venv.log")
    monkeypatch.setattr(api.runner, "tail_log",
                        lambda path, tail_bytes=20000: "VENV_DONE rc=0\n")
    for _ in range(200):
        if j.state in ("done", "failed"):
            break
        time.sleep(0.05)
    assert j.state == "done"


def test_eval_env_install(client, monkeypatch):
    monkeypatch.setattr(api.runner, "ensure_eval_env", lambda log, cfg: None)
    import server.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "_sleep", lambda s: time.sleep(0.01))
    r = client.post("/api/eval-env")
    assert r.status_code == 202
    j = api.REGISTRY.get(r.json()["job_id"])
    assert j is not None
    assert j.kind == "eval_env"
    for _ in range(200):
        if j.state in ("done", "failed"):
            break
        time.sleep(0.05)
    assert j.state == "done"


def test_abort_job(client):
    j = api.REGISTRY.create("venv", None, "x")
    r = client.post(f"/api/jobs/{j.id}/abort")
    assert r.status_code == 200
    assert j.state == "aborted"
    assert r.json()["state"] == "aborted"


def test_abort_job_404(client):
    r = client.post("/api/jobs/nope_123/abort")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "no_job"


def test_recipe_diff_terminal_job_without_task(client, monkeypatch, tmp_path):
    """服务重启后终态 job 的 task 被清空（job.json 持久化）：对照必须走
    持久化字段 job.cmd / job.log_path，而不是 job.task（旧版直接 400）。"""
    from webui import validate
    j = api.REGISTRY.create("quant", None,
                            "cd /repo && /venv/bin/python hf_ptq.py --model x",
                            )
    j.log_path = str(tmp_path / "r.log")
    j.state = "done"
    cfg_json = json.dumps({"quantization_config": {"quant_method": "modelopt"}})
    monkeypatch.setattr(api.runner, "tail_log", lambda path, n: "exported to: exported_model")
    monkeypatch.setattr(validate, "run_shell",
                        lambda cmd, timeout=60: (0, cfg_json, ""))

    r = client.post("/api/recipe/diff", json={"job_id": j.id, "official": ""})
    assert r.status_code == 200
    assert r.json()["summary"]["source"] == "本机导出 /repo/exported_model"


def test_recipe_diff_no_job(client):
    r = client.post("/api/recipe/diff", json={"job_id": "nope_123"})
    assert r.status_code == 404


def test_recipe_diff_passes_bare_proxy_url(client, monkeypatch, tmp_path):
    """回归：官方对照拿到的 proxy 必须是纯 URL。

    旧 bug：把 proxy_for()（shell export 前缀串，拼 nohup 命令用的）当 URL 传给
    curl，http_proxy 被设成整串「export ...」→ curl exit 5 连不上代理，而网络
    面板测试（走 proxy_url_for）却是过的——两边不一致最能骗人。"""
    from webui import validate
    api.CFG.net.mode = "manual"
    api.CFG.net.proxy = "192.168.124.24:7890"
    j = api.REGISTRY.create("quant", None, "cd /repo && /venv/bin/python hf_ptq.py --model x")
    j.log_path = str(tmp_path / "r.log")
    j.state = "done"
    cfg_json = json.dumps({"quantization_config": {"quant_method": "modelopt"}})
    monkeypatch.setattr(api.runner, "tail_log", lambda path, n: "exported to: exported_model")
    monkeypatch.setattr(validate, "run_shell", lambda cmd, timeout=60: (0, cfg_json, ""))
    seen = {}

    def fake_official(model_id, proxy=""):
        seen["proxy"] = proxy
        return validate.QuantSummary(source=model_id)

    monkeypatch.setattr(validate, "fetch_official_summary", fake_official)
    r = client.post("/api/recipe/diff", json={"job_id": j.id, "official": "nvidia/Fake-NVFP4"})
    assert r.status_code == 200
    assert seen["proxy"] == "http://192.168.124.24:7890"
    assert "export " not in seen["proxy"]


def test_recipe_diff_bad_job(client):
    j = api.REGISTRY.create("venv", None, "x")
    r = client.post("/api/recipe/diff", json={"job_id": j.id})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_job"


def test_recipe_diff_relative_export_resolves_to_cd_dir(client, monkeypatch, tmp_path):
    """cmd 无 --export_path 时：日志里的相对导出路径（exported_model）必须拼到
    cmd 里 `cd <repo>` 的目录下，而不是回退成第一个词（旧 bug：hint='cd' →
    读 cd/config.json 必败）。"""
    from webui import validate
    task = runner_mod.Task(task_id="t", remote_cmd="cd /repo && /venv/bin/python hf_ptq.py --model x",
                           log_path=str(tmp_path / "r.log"), pid="1", state="done")
    j = api.REGISTRY.create("quant", task, task.remote_cmd)
    monkeypatch.setattr(api.runner, "tail_log",
                        lambda path, n: "Quantized model exported to: exported_model. done")
    cfg_json = json.dumps({"quantization_config": {
        "quant_method": "modelopt", "quant_algo": "NVFP4"}})
    seen = []

    def fake_run_shell(cmd, timeout=60):
        seen.append(cmd)
        if cmd.startswith("cat "):
            return (0, cfg_json, "")
        return (0, "files", "")
    monkeypatch.setattr(validate, "run_shell", fake_run_shell)

    r = client.post("/api/recipe/diff", json={"job_id": j.id, "official": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["quant_method"] == "modelopt"
    assert body["summary"]["source"] == "本机导出 /repo/exported_model"
    assert any(cmd.startswith("cat /repo/exported_model/config.json") for cmd in seen)


def test_recipe_diff_export_path_flag_wins(client, monkeypatch, tmp_path):
    """cmd 有 --export_path 时，无日志 marker 也应回退到该绝对路径（不再读 cd/config.json）。"""
    from webui import validate
    task = runner_mod.Task(task_id="t",
                           remote_cmd="cd /repo && py hf_ptq.py --export_path /out/q4",
                           log_path=str(tmp_path / "r.log"), pid="1", state="done")
    j = api.REGISTRY.create("quant", task, task.remote_cmd)
    monkeypatch.setattr(api.runner, "tail_log", lambda path, n: "no marker here")
    cfg_json = json.dumps({"quantization_config": {"quant_method": "modelopt"}})
    seen = []
    monkeypatch.setattr(validate, "run_shell",
                        lambda cmd, timeout=60: (seen.append(cmd) or (0, cfg_json, "")))

    r = client.post("/api/recipe/diff", json={"job_id": j.id, "official": ""})
    assert r.status_code == 200
    assert r.json()["summary"]["source"] == "本机导出 /out/q4"
    assert any(cmd.startswith("cat /out/q4/config.json") for cmd in seen)


def test_eval_baseline_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api.runner, "run_shell", lambda cmd, timeout=60: (1, "", ""))
    r = client.post("/api/eval/baseline", json={"quant_path": str(tmp_path / "nope")})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_quant_path"


def test_eval_baseline_ok(client, monkeypatch, tmp_path):
    cfg_json = json.dumps({"quantization_config": {"_name_or_path": "nvidia/Org-7B"}})
    monkeypatch.setattr(api.runner, "run_shell",
                        lambda cmd, timeout=60: (0, cfg_json, ""))
    r = client.post("/api/eval/baseline", json={"quant_path": str(tmp_path / "q")})
    assert r.status_code == 200
    assert r.json()["baseline"] == "nvidia/Org-7B"


def test_put_paths_and_net(client, monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.yaml"
    # save_local 无 path 参数，直接写项目根 config.yaml —— 测试里重定向到 tmp
    monkeypatch.setattr(api, "save_local",
                        lambda local: cfg_file.write_text(
                            f"repo: {local.repo}\nvenv: {local.venv}\n"
                            f"task_dir: {local.task_dir}\nhf_cache: {local.hf_cache}\n"))
    r1 = client.put("/api/paths", json={
        "repo": str(tmp_path / "repo"), "venv": str(tmp_path / "v"),
        "task_dir": str(tmp_path / "t"), "hf_cache": str(tmp_path / "h")})
    assert r1.status_code == 200
    assert api.CFG.local.repo == str(tmp_path / "repo")   # 内存态已更新
    assert cfg_file.exists()   # save_local 被调到
    # save_net 指向临时文件（避免污染项目根 net.json）
    net_file = tmp_path / "net.json"
    monkeypatch.setattr(api, "save_net",
                        lambda net, path=net_file: path.write_text(json.dumps(
                            {"mode": net.mode, "proxy": net.proxy})))
    r2 = client.put("/api/net", json={"mode": "manual", "proxy": "http://x:1"})
    assert r2.status_code == 200
    assert net_file.exists()
    assert json.loads(net_file.read_text())["mode"] == "manual"


def test_put_paths_empty_falls_back_to_defaults(client, monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.yaml"
    monkeypatch.setattr(api, "save_local",
                        lambda local: cfg_file.write_text("{}"))
    r = client.put("/api/paths", json={"repo": "", "venv": "", "task_dir": "", "hf_cache": ""})
    assert r.status_code == 200
    assert (api.CFG.local.repo, api.CFG.local.venv,
            api.CFG.local.task_dir, api.CFG.local.hf_cache) == \
        (DEFAULT_REPO, DEFAULT_VENV, DEFAULT_TASK_DIR, DEFAULT_HF_CACHE)


def test_put_paths_save_failed(client, monkeypatch):
    def _boom(local):
        raise OSError("disk full")
    monkeypatch.setattr(api, "save_local", _boom)
    r = client.put("/api/paths", json={
        "repo": "/a", "venv": "/b", "task_dir": "/c", "hf_cache": "/d"})
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "save_failed"


def test_net_test_failed(client, monkeypatch):
    def _boom(cfg, url=""):
        raise TimeoutError("代理超时")
    monkeypatch.setattr(api.runner, "test_proxy", _boom)
    r = client.post("/api/net/test")
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "test_failed"


def test_net_test(client, monkeypatch):
    monkeypatch.setattr(api.runner, "test_proxy", lambda cfg, url="": (True, "HTTP 200"))
    r = client.post("/api/net/test")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "detail": "HTTP 200"}


def test_net_hf_token_semantics(client, monkeypatch, tmp_path):
    """HF_TOKEN：值永不回显；省略=保持、""=清空、非空=替换（对照 gated 数据集需求）。

    假令牌值用拼接构造：否则会被 secret 扫描规则当硬编码凭据拦（误报）。"""
    tok_a, tok_b = "TK-" + "ALPHA", "TK-" + "BETA"
    net_file = tmp_path / "net.json"
    monkeypatch.setattr(api, "save_net", lambda net, path=net_file: path.write_text(
        json.dumps({"network": {"mode": net.mode, "proxy": net.proxy,
                                "hf_token": net.hf_token}})))
    body = {"mode": "manual", "proxy": "http://x:1", "hf_token": tok_a}
    assert client.put("/api/net", json=body).status_code == 200
    assert api.CFG.net.hf_token == tok_a
    g = client.get("/api/net").json()["net"]
    assert g["hf_token_set"] is True
    assert tok_a not in json.dumps(g)                        # 不外泄
    client.put("/api/net", json={"mode": "auto", "proxy": ""})
    assert api.CFG.net.hf_token == tok_a                     # 省略字段 = 保持
    body2 = {"mode": "auto", "proxy": "", "hf_token": tok_b}
    client.put("/api/net", json=body2)
    assert api.CFG.net.hf_token == tok_b                     # 非空替换
    body3 = {"mode": "auto", "proxy": "", "hf_token": ""}
    client.put("/api/net", json=body3)
    assert api.CFG.net.hf_token == ""                        # 空串清空


def test_net_json_hf_token_roundtrip(tmp_path):
    from webui.config import NetConfig, load_net, save_net
    tok = "TK-" + "ROUND"
    p = tmp_path / "net.json"
    nc = NetConfig(mode="manual", proxy="http://p:1")
    nc.hf_token = tok
    save_net(nc, path=p)
    assert load_net(path=p).hf_token == tok
    p.write_text('{"network": {"mode": "auto", "proxy": ""}}')   # 旧文件无该键
    assert load_net(path=p).hf_token == ""


def test_datasets_list_exposes_gated(client):
    """gated 数据集标记要能到前端（nemotron v2 是唯一 gated 的内置数据集）。"""
    ds = {d["key"]: d for d in api._datasets_list()}
    assert ds["nemotron-post-training-dataset-v2"]["gated"] is True
    assert ds["cnn_dailymail"]["gated"] is False

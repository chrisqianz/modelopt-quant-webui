"""server/api.py 只读端点冒烟测试（TestClient + monkeypatch，不真跑量化）。

本机无 CUDA：所有 webui.runner 探测/日志/结果函数与 server.capabilities
的探测都 patch 掉；jobs 单例直接调 REGISTRY.create 造数据。
"""

import json

import pytest
from fastapi.testclient import TestClient

import server.api as api
from server.jobs import JobRegistry
from webui import runner as runner_mod
from webui import vocab as vocab_mod
from webui.config import AppConfig, LocalConfig, NetConfig
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
                                   "--calib_size", "--kv_cache_qformat",
                                   "--attn_implementation", "--export_path"],
                    "mapped": ["--qformat"], "unmapped": [], "missing": [], "error": ""})
    return TestClient(api.app)


# ---------------------------------------------------------------- 端点形状

def test_bootstrap_shape(client):
    r = client.get("/api/bootstrap")
    assert r.status_code == 200
    body = r.json()
    for k in ("paths", "net", "probe", "vocab", "capabilities", "jobs", "defaults"):
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
    assert all(set(d) == {"key", "desc"} for d in body["datasets"])
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
    assert "\n" in body["pretty"]   # 多行展示形态


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
    assert r.json()["net"] == {"mode": "auto", "proxy": "http://127.0.0.1:7890"}


# ---------------------------------------------------------------- 本地模型 / 产物

def test_models_local(client, tmp_path):
    hub = tmp_path / "hf" / "hub"
    (hub / "models--org--name" / "snapshots" / "s1").mkdir(parents=True)
    (hub / "models--org--stub").mkdir(parents=True)
    r = client.get("/api/models/local")
    assert r.status_code == 200
    body = r.json()
    assert body["error"] == ""
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["org/name"]["stub"] is False
    assert by_id["org/name"]["size"] == "1.2G"
    assert by_id["org/stub"]["stub"] is True


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
    assert out[0]["size"] == "1.2G"


def test_artifacts_with_job_json(client, tmp_path):
    tasks = tmp_path / "tasks"
    d = tasks / "eval_y_20240202"
    d.mkdir(parents=True)
    (d / "run.log").write_text("log\n")
    (d / "job.json").write_text(json.dumps({"state": "done", "export_dir": str(tmp_path / "q"), "started_at": 1700000000}))
    r = client.get("/api/artifacts")
    out = r.json()["artifacts"]
    assert len(out) == 1
    assert out[0]["state"] == "done"
    assert out[0]["model"] == str(tmp_path / "q")
    assert out[0]["ts"] == "1700000000"


def test_artifacts_empty_dir(client):
    r = client.get("/api/artifacts")
    assert r.status_code == 200
    assert r.json() == {"artifacts": []}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

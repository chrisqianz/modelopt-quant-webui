"""显存估算器单测（mock du / HF API / probe_host，全路径）。

真实磁盘探测（du、config.json 读取、nvidia-smi）在 GPU 机验收里做；
这里用 monkeypatch 把外部 IO 换成确定值，专测估算公式与 source 标注。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import server.api as api
import server.estimate as est
from webui.runner import HostStatus


def _make_cfg(tmp_path):
    from webui.config import AppConfig, LocalConfig, NetConfig
    return AppConfig(
        server_host="127.0.0.1", server_port=7861, server_token="",
        local=LocalConfig(repo=str(tmp_path / "repo"), venv=str(tmp_path / "venv"),
                          task_dir=str(tmp_path / "tasks"), hf_cache=str(tmp_path / "hf")),
        net=NetConfig(mode="auto", proxy=""),
        defaults={},
    )


def _mkdir_cache(tmp_path, model="org/name"):
    """造 hub 缓存目录，让 _weight_bytes 走 cached.exists() 分支。"""
    d = tmp_path / "hf" / "hub" / ("models--" + model.replace("/", "--", 1))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_model_cfg(tmp_path, model="org/name", layers=12, kv_heads=6,
                     head_dim=None, hidden_size=None, n_heads=None):
    """造 hub/models--org--name/snapshots/rev/config.json。"""
    _mkdir_cache(tmp_path, model)
    snap = tmp_path / "hf" / "hub" / "models--org--name" / "snapshots" / "rev1"
    snap.mkdir(parents=True, exist_ok=True)
    import json
    data = {"num_hidden_layers": layers, "num_key_value_heads": kv_heads}
    if head_dim:
        data["head_dim"] = head_dim
    if hidden_size:
        data["hidden_size"] = hidden_size
    if n_heads:
        data["num_attention_heads"] = n_heads
    (snap / "config.json").write_text(json.dumps(data))


def test_hf_id_cache_hit_source_cache(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _mkdir_cache(tmp_path)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 4 * 1024 ** 3)  # 4GB 缓存
    monkeypatch.setattr(est, "_hf_api_size", lambda m, c: (0, False))
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))   # 64GB 全空闲
    r = est.estimate("org/name", "nvfp4", "none", 2048, 1, cfg)
    assert r["weight_bytes"] == 4 * 1024 ** 3
    assert r["source"] == "cache"
    assert r["fits"] is True


def test_hf_id_cache_miss_falls_to_api(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    # 缓存目录存在但 du 返 0 ⇒ 走 HF API
    _mkdir_cache(tmp_path)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 0)             # 缓存没有
    monkeypatch.setattr(est, "_hf_api_size", lambda m, c: (8 * 1024 ** 3, True))
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))
    r = est.estimate("org/name", "fp8", "none", 2048, 1, cfg)
    assert r["source"] == "hf_api"
    assert r["weight_bytes"] == 8 * 1024 ** 3
    # fp8 系数 1.0
    assert r["quant_bytes"] == 8 * 1024 ** 3


def test_unknown_when_both_fail(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _mkdir_cache(tmp_path)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 0)
    monkeypatch.setattr(est, "_hf_api_size", lambda m, c: (0, False))
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))
    r = est.estimate("org/name", "nvfp4", "none", 2048, 1, cfg)
    assert r["weight_bytes"] == 0
    assert r["source"] == "unknown"


def test_local_path_du(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    (tmp_path / "abs" / "model").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 2 * 1024 ** 3)
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))
    r = est.estimate(str(tmp_path / "abs" / "model"), "nvfp4", "none", 2048, 1, cfg)
    assert r["source"] == "cache"
    assert r["weight_bytes"] == 2 * 1024 ** 3


def test_kv_from_config_json(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _write_model_cfg(tmp_path, layers=12, kv_heads=6, head_dim=128)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 8 * 1024 ** 3)  # cache 命中
    monkeypatch.setattr(est, "_hf_api_size", lambda m, c: (0, False))
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))
    # nvfp4 kv elem=0.5 → 2*12*6*128*2048*1*0.5
    expect_kv = int(2 * 12 * 6 * 128 * 2048 * 1 * 0.5)
    r = est.estimate("org/name", "nvfp4", "nvfp4", 2048, 1, cfg)
    assert r["kv_bytes"] == expect_kv
    assert "kv_estimate" not in r["source"]


def test_kv_fallback_estimate_when_no_config(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _mkdir_cache(tmp_path)  # 缓存命中但无 config.json ⇒ _kv_from_config 返回 None
    monkeypatch.setattr(est, "_du_bytes", lambda d: 8 * 1024 ** 3)
    monkeypatch.setattr(est, "_hf_api_size", lambda m, c: (0, False))
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))
    r = est.estimate("org/name", "nvfp4", "nvfp4", 2048, 2, cfg)
    # 经验公式: weight * 0.15 * ctx/2048 * batch
    expect = int(8 * 1024 ** 3 * 0.15 * (2048 / 2048) * 2)
    assert r["kv_bytes"] == expect
    assert "kv_estimate" in r["source"]
    assert r["note"]


def test_kv_none_zero(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _write_model_cfg(tmp_path, layers=12, kv_heads=6, head_dim=128)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 8 * 1024 ** 3)  # cache 命中
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))
    r = est.estimate("org/name", "nvfp4", "none", 2048, 1, cfg)
    assert r["kv_bytes"] == 0


def test_does_not_fit_short_gib(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _mkdir_cache(tmp_path)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 60 * 1024 ** 3)  # 60GB 权重
    monkeypatch.setattr(est, "_hf_api_size", lambda m, c: (0, False))
    monkeypatch.setattr(est, "_vram_split", lambda c: (4096, 24576))  # 24GB 总量
    r = est.estimate("org/name", "nvfp4", "none", 2048, 1, cfg)
    assert r["fits"] is False
    assert r["short_gib"] > 0
    assert r["vram_total_mib"] == 24576
    assert r["vram_used_mib"] == 4096


def test_quant_factor_table(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _mkdir_cache(tmp_path)
    monkeypatch.setattr(est, "_du_bytes", lambda d: 1000)
    monkeypatch.setattr(est, "_vram_split", lambda c: (0, 65536))
    base = est.estimate("org/name", "nvfp4", "none", 2048, 1, cfg)
    assert base["quant_bytes"] == int(1000 * 0.55)
    fp8 = est.estimate("org/name", "fp8", "none", 2048, 1, cfg)
    assert fp8["quant_bytes"] == 1000


def test_api_endpoint_400_on_empty_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CFG", _make_cfg(tmp_path))
    monkeypatch.setattr(api.runner, "probe_host",
                        lambda cfg: HostStatus(reachable=True, gpu_mem="3/24576 MiB"))
    client = TestClient(api.app)
    r = client.get("/api/estimate")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_params"


def test_api_endpoint_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CFG", _make_cfg(tmp_path))
    monkeypatch.setattr(api.runner, "probe_host",
                        lambda cfg: HostStatus(reachable=True, gpu_mem="3/24576 MiB"))
    monkeypatch.setattr(est, "_du_bytes", lambda d: 0)
    monkeypatch.setattr(est, "_hf_api_size", lambda m, c: (0, False))
    client = TestClient(api.app)
    r = client.get("/api/estimate?model=org%2Fname&qformat=nvfp4&kv_qformat=none&ctx=2048&batch=1")
    assert r.status_code == 200
    assert r.json()["source"] == "unknown"

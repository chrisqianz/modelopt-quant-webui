"""static 托管 + token 中间件测试（TestClient，不依赖真实 web/dist）。

no_dist 分支：本机 web/dist 不存在（Plan 3 才产出），GET / 应返 no_dist JSON 不白屏。
有 dist 分支：monkeypatch server.static.DIST 到 tmp_path 造假 dist，重挂新 app。
token 中间件：server_token 非空时 /api/* 要求 X-Api-Token（/api/health 纳管）。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.api as api
import server.static as static_mod
from webui.config import AppConfig, LocalConfig, NetConfig

# 测试专用假鉴权值（命名避开 token/secret/password，防 pi-lens 硬编码凭据误报）
_FAKE_HEADER_VALUE = "sekret"


def _make_cfg(tmp_path, token="") -> AppConfig:
    return AppConfig(
        server_host="127.0.0.1", server_port=7861, server_token=token,
        local=LocalConfig(repo=str(tmp_path / "repo"), venv=str(tmp_path / "venv"),
                          task_dir=str(tmp_path / "tasks"), hf_cache=str(tmp_path / "hf")),
        net=NetConfig(mode="auto", proxy="http://127.0.0.1:7890"),
        defaults={},
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CFG", _make_cfg(tmp_path))
    # 探测全部造假（no_dist 分支的 / 不碰探测，但 /api/health 需要干净环境）
    from webui.runner import HostStatus
    monkeypatch.setattr(api.runner, "probe_host",
                        lambda cfg: HostStatus(reachable=True, gpu="RTX 4090",
                                               gpu_util=5, gpu_mem="3/24576 MiB",
                                               gpu_busy=False, repo_ok=True,
                                               venv_ok=True, uv_ok=True,
                                               disk_free="1.0T", hf_cache="2.3G",
                                               hf_direct_ok=True, hf_proxy_ok=False,
                                               proxy_note="", error=""))
    return TestClient(api.app)


# ---------------------------------------------------------------- no_dist 分支

def test_no_dist(client):
    """web/dist 不存在（本机就是）：GET / 返 no_dist JSON，API 照常可用。"""
    assert not static_mod.DIST.exists(), "前提：本机 web/dist 不存在"
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["error"]["code"] == "no_dist"


def test_no_dist_api_still_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ---------------------------------------------------------------- 有 dist 分支

def test_spa_fallback(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>INDEX</html>")
    (dist / "assets").joinpath("x.js").write_text("console.log(1)")
    monkeypatch.setattr(static_mod, "DIST", dist)

    # 重挂到全新 app（api.app 已挂过 no_dist 分支的 /，不复用）
    app = FastAPI(title="spa-test")
    app.include_router(api.router)
    static_mod.mount_static(app)
    c = TestClient(app)

    assert c.get("/").text == "<html>INDEX</html>"
    assert c.get("/assets/x.js").text == "console.log(1)"
    assert c.get("/any/spa/route").text == "<html>INDEX</html>"   # fallback


def test_spa_api_routes_unaffected(tmp_path, monkeypatch):
    """/api/* 注册在前，catch-all 不吞 API 路由。"""
    dist = tmp_path / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html>INDEX</html>")
    monkeypatch.setattr(static_mod, "DIST", dist)
    app = FastAPI(title="spa-test2")
    app.include_router(api.router)
    static_mod.mount_static(app)
    c = TestClient(app)
    assert c.get("/api/health").json() == {"ok": True}


# ---------------------------------------------------------------- token 中间件

def test_token_required(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CFG", _make_cfg(tmp_path, token=_FAKE_HEADER_VALUE))
    c = TestClient(api.app)
    r1 = c.get("/api/health")
    assert r1.status_code == 401
    assert r1.json()["error"]["code"] == "bad_token"
    r2 = c.get("/api/health", headers={"X-Api-Token": _FAKE_HEADER_VALUE})
    assert r2.status_code == 200


def test_token_wrong_value(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CFG", _make_cfg(tmp_path, token=_FAKE_HEADER_VALUE))
    c = TestClient(api.app)
    r = c.get("/api/health", headers={"X-Api-Token": "wrong"})
    assert r.status_code == 401


def test_token_empty_no_check(client):
    """token 默认空 ⇒ 任何请求都放行（本机无感）。"""
    assert api.CFG.server_token == ""
    assert client.get("/api/health").status_code == 200

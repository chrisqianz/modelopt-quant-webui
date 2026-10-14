"""tests/server/ 共享 fixtures：临时路径下的 AppConfig + JobRegistry。"""
import pytest

from server.jobs import JobRegistry
from webui.config import AppConfig, LocalConfig, NetConfig


@pytest.fixture
def cfg(tmp_path) -> AppConfig:
    return AppConfig(
        server_host="127.0.0.1", server_port=7861, server_token="",
        local=LocalConfig(repo=str(tmp_path / "repo"), venv=str(tmp_path / "venv"),
                          task_dir=str(tmp_path / "tasks"), hf_cache=str(tmp_path / "hf")),
        net=NetConfig(mode="auto", proxy=""),
        defaults={},
    )


@pytest.fixture
def registry(cfg) -> JobRegistry:
    return JobRegistry(cfg)

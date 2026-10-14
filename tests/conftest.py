import pytest


@pytest.fixture(autouse=True)
def _pin_home(monkeypatch):
    """把 HOME 钉死：命令串里的 `~` 会被 q_remote 展开成绝对路径，
    不钉住就会把「跑测试这台机器」烘进 golden —— 在 mac 上生成的快照
    拿到 GPU 机（HOME=/home/chrisqian）上必然对不上。"""
    monkeypatch.setenv("HOME", "/home/tester")


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="重写 golden/*.json（只在故意改行为时用，改完看 diff）")

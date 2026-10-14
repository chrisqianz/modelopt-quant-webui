"""server/capabilities 的最小契约测试（Task 3）。

静态探测本机 Model-Optimizer 仓库支持哪些 hf_ptq flag，UI flag 与仓库 flag
对账（mapped / unmapped / missing）。用 tmp_path 造假仓库，monkeypatch
runner.run_shell 返回假的 git describe / rev-parse 结果。
"""
from server.capabilities import (
    FLAG_UI,
    extract_flags,
    probe_capabilities,
    validate_cmd_flags,
)
from webui import runner
from webui.config import AppConfig, LocalConfig

FAKE_REPO_SRC = (
    'parser.add_argument("--qformat", type=str)\n'
    'parser.add_argument("--calib_seq", type=int)\n'
    'parser.add_argument("--foo_new_flag", type=str)\n'
    'parser.add_argument("--skip_generate", type=bool)\n'
)


def _make_fake_repo(tmp_path) -> None:
    (tmp_path / "examples" / "hf_ptq").mkdir(parents=True)
    (tmp_path / "examples/hf_ptq/hf_ptq.py").write_text(FAKE_REPO_SRC)


def _cfg(repo) -> AppConfig:
    return AppConfig(local=LocalConfig(repo=str(repo)))


def _fake_run_shell(cmd: str, timeout: int = 60):
    """按命令分发假的 git 结果。"""
    if "describe --tags" in cmd:
        return (0, "v0.48.0\n", "")
    if "rev-parse --short HEAD" in cmd:
        return (0, "abc1234\n", "")
    return (0, "", "")


# ---------------------------------------------------------------- extract_flags

def test_extract_flags_finds_all_no_dups(tmp_path):
    _make_fake_repo(tmp_path)
    flags = extract_flags(tmp_path)
    for f in ("--qformat", "--calib_seq", "--foo_new_flag", "--skip_generate"):
        assert f in flags
    assert flags == ["--qformat", "--calib_seq", "--foo_new_flag", "--skip_generate"]


def test_extract_flags_missing_file_returns_empty(tmp_path):
    # 空仓库：没有 examples/hf_ptq/hf_ptq.py
    assert extract_flags(tmp_path) == []


# ---------------------------------------------------------------- probe_capabilities

def test_probe_capabilities_happy_path(tmp_path, monkeypatch):
    _make_fake_repo(tmp_path)
    monkeypatch.setattr(runner, "run_shell", _fake_run_shell)
    out = probe_capabilities(_cfg(tmp_path))

    assert out["error"] == ""
    assert out["version"] == "v0.48.0"
    assert out["commit"] == "abc1234"
    assert out["flags_all"] == ["--qformat", "--calib_seq",
                                "--foo_new_flag", "--skip_generate"]
    # mapped = FLAG_UI 的值 ∩ flags_all（按 FLAG_UI 顺序）：--calib_seq 在 --skip_generate 前
    assert out["mapped"] == ["--qformat", "--calib_seq", "--skip_generate"]
    # unmapped = 仓库有但 UI 不拼的
    assert out["unmapped"] == ["--foo_new_flag"]
    # missing = UI 要拼但仓库没有的（假仓库没有 --pyt_ckpt_path 等）
    assert "--pyt_ckpt_path" in out["missing"]
    # missing 覆盖全部 18 项里假仓库缺的那部分
    fake = set(out["flags_all"])
    expected_missing = [v for v in FLAG_UI.values() if v not in fake]
    assert out["missing"] == expected_missing


def test_probe_capabilities_missing_file_sets_flags_empty(tmp_path, monkeypatch):
    # 空仓库：flag 全缺，但版本/commit 仍能取到
    monkeypatch.setattr(runner, "run_shell", _fake_run_shell)
    out = probe_capabilities(_cfg(tmp_path))
    assert out["flags_all"] == []
    assert out["mapped"] == []
    assert out["unmapped"] == []
    assert out["missing"] == list(FLAG_UI.values())


def test_probe_capabilities_error_on_exception(tmp_path, monkeypatch):
    def boom(cmd, timeout=60):
        raise RuntimeError("git not found")
    monkeypatch.setattr(runner, "run_shell", boom)
    out = probe_capabilities(_cfg(tmp_path))
    assert out["error"] != ""
    assert "git not found" in out["error"]


# ---------------------------------------------------------------- validate_cmd_flags

def test_validate_cmd_flags_returns_unknown():
    assert validate_cmd_flags(
        ".venv/bin/python hf_ptq.py --qformat nvfp4 --bad_flag x",
        ["--qformat"],
    ) == ["--bad_flag"]


def test_validate_cmd_flags_all_known():
    assert validate_cmd_flags(
        "python hf_ptq.py --qformat nvfp4 --calib_seq 512",
        ["--qformat", "--calib_seq"],
    ) == []


def test_validate_cmd_flags_dedupes():
    assert validate_cmd_flags(
        "a --nope b --nope c",
        [],
    ) == ["--nope"]


def test_validate_cmd_flags_ignores_non_flag_dash():
    # 不是 -- 前缀的（如 -h 或 2>/dev/null 里的 >）不算
    assert validate_cmd_flags("python -h x", []) == []

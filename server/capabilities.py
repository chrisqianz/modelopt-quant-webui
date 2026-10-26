"""版本耦合应对第一层：静态探测本机 Model-Optimizer 仓库支持哪些 hf_ptq flag。

不跑 ``python hf_ptq.py --help``（那要 import torch），只做 ``re`` 静态抽取
``add_argument`` 字符串，然后把 UI flag 与仓库 flag 对账（mapped / unmapped /
missing）。``missing`` 供发射前拦截：UI 要拼但仓库没有的 flag。
"""

from __future__ import annotations

import re
from pathlib import Path

from webui import runner
from webui.config import AppConfig

# ---------------------------------------------------------------- 映射表

# UI 字段名 -> hf_ptq.py flag 名（19 项，逐项对照 webui/cmds.py build_cmd 的
# parts.append 真实字面量）。UI 字段名即 build_params 的 dict key。
FLAG_UI: dict[str, str] = {
    "model": "--pyt_ckpt_path",
    "trust_remote_code": "--trust_remote_code",
    "recipe": "--recipe",
    "qformat": "--qformat",
    "kv_cache_qformat": "--kv_cache_qformat",
    "datasets": "--dataset",
    "dataset_sizes": "--calib_size",
    "calib_seq": "--calib_seq",
    "calib_with_images": "--calib_with_images",
    "export_path": "--export_path",
    "batch_size": "--batch_size",
    "device": "--device",
    "attn_implementation": "--attn_implementation",
    "use_fsdp2": "--use_fsdp2",
    "cpu_offload": "--cpu_offload",
    "low_memory_mode": "--low_memory_mode",
    "gpu_max_mem_percentage": "--gpu_max_mem_percentage",
    "skip_generate": "--skip_generate",
    "use_seq_device_map": "--use_seq_device_map",
}

_HF_PTQ_RELPATH = ("examples", "hf_ptq", "hf_ptq.py")


# ---------------------------------------------------------------- 静态抽取

def extract_flags(repo: Path) -> list[str]:
    """静态抽取 hf_ptq.py 里所有 add_argument 的 ``--xxx`` flag，去重保序。

    文件不存在时返回 []（probe_capabilities 层面给 error）。
    """
    src_path = Path(repo).joinpath(*_HF_PTQ_RELPATH)
    if not src_path.exists():
        return []
    try:
        src = src_path.read_text()
    except OSError:
        return []
    found = re.findall(r"add_argument\s*\(\s*['\"](--[a-zA-Z0-9_]+)", src)
    return list(dict.fromkeys(found))


def probe_capabilities(cfg: AppConfig) -> dict:
    """版本（git describe / rev-parse）+ flag 对账。异常时 error 非空。"""
    out: dict = {
        "version": "unknown",
        "commit": "unknown",
        "flags_all": [],
        "mapped": [],
        "unmapped": [],
        "missing": [],
        "error": "",
    }
    repo = cfg.local.repo_path   # 走 resolve_relative：expanduser + 相对路径解析
    # （裸 Path(cfg.local.repo) 不展开 ~，extract_flags/exists 会静默失效；
    #   git -C 再经 q_remote 转义，与项目惯例一致，避免 repo 含特殊字符时经 bash -c 注入）
    repo_q = runner.q_remote(str(repo))
    try:
        rc, out_v, _ = runner.run_shell(
            f"git -C {repo_q} describe --tags --always 2>/dev/null || echo unknown")
        out["version"] = out_v.strip() or "unknown"
        rc, out_c, _ = runner.run_shell(
            f"git -C {repo_q} rev-parse --short HEAD 2>/dev/null || echo unknown")
        out["commit"] = out_c.strip() or "unknown"
        flags_all = extract_flags(repo)
        out["flags_all"] = flags_all
        if not flags_all and not repo.joinpath(*_HF_PTQ_RELPATH).exists():
            out["error"] = f"hf_ptq.py 不存在: {repo.joinpath(*_HF_PTQ_RELPATH)}"
        ui_values = list(FLAG_UI.values())
        out["mapped"] = [f for f in ui_values if f in flags_all]
        out["unmapped"] = [f for f in flags_all if f not in ui_values]
        out["missing"] = [f for f in ui_values if f not in flags_all]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def validate_cmd_flags(cmd: str, flags_all: list[str]) -> list[str]:
    """抽命令串里所有 ``--xxx`` token，返回不在 flags_all 里的（去重保序）。

    发射前差集校验：非空即命令含仓库不认识的 flag。
    """
    found = re.findall(r"(?<!\w)--[a-zA-Z0-9_]+", cmd)
    allowed = set(flags_all)
    return [f for f in dict.fromkeys(found) if f not in allowed]

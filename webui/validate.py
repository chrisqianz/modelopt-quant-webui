"""结果校验：量化完成后核对导出 checkpoint 的实际量化范围。

数据来源两条，都不占 GPU：
1. 本机导出目录的 config.json（cat + json 解析）
2. 官方模型卡的 quantization_config（走代理访问 HF API，用于对照）

输出统一的"量化范围摘要"：groups（位数 × target 数 × 代表 target）+ ignore 列表，
并与官方对照给出 一致/差异 提示。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .runner import run_shell

_HF_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


@dataclass
class QuantSummary:
    source: str = ""  # e.g. "导出目录" / "nvidia/Qwen3.8-27B-NVFP4"
    quant_method: str = ""
    groups: list[dict[str, Any]] = field(default_factory=list)  # {bits, count, samples}
    ignore: list[str] = field(default_factory=list)
    files: str = ""  # 导出目录文件清单（仅本机导出摘要）
    error: str = ""


def summarize(qc: dict[str, Any], source: str = "") -> QuantSummary:
    """把 quantization_config（NVIDIA 格式）压成可展示的摘要。"""
    s = QuantSummary(source=source, quant_method=qc.get("quant_method", ""))
    for name, g in (qc.get("config_groups") or {}).items():
        w = g.get("weights", {})
        targets = g.get("targets", []) or []
        s.groups.append({
            "bits": w.get("num_bits", "?"),
            "count": len(targets),
            "samples": targets[:5],
            "more": f"…共 {len(targets)} 个" if len(targets) > 5 else "",
            "name": name,
        })
    s.ignore = qc.get("ignore", []) or []
    return s


# ---------------------------------------------------------------- 本机导出目录

def scan_export_dir(log_tail: str, export_dir_hint: str = "") -> QuantSummary:
    """从任务日志定位导出目录，读本机 config.json 并摘要。"""
    m = re.search(r"Quantized model exported to:\s*(\S+)", log_tail)
    target = (m.group(1) if m else "").rstrip(".")
    if not target:
        target = export_dir_hint
    if not target:
        return QuantSummary(error="日志中未找到 'Quantized model exported to:'，无法定位导出目录")

    q = shlex.quote(target)
    r, out, err = run_shell(f"cat {q}/config.json 2>/dev/null", timeout=20)
    if r != 0 or not out:
        return QuantSummary(error=f"读取 {target}/config.json 失败（{err or '文件不存在'}）")
    try:
        cfg = json.loads(out)
    except json.JSONDecodeError as ex:
        return QuantSummary(error=f"config.json 解析失败: {ex}")
    s = summarize(cfg.get("quantization_config", {}), source=f"本机导出 {target}")
    # 附带文件清单，验证导出完整性（保持 ls -lh 原文口径，UI 表格直接显示）
    r2, files, _ = run_shell(f"ls -lh {q} 2>/dev/null | awk '{{print $5, $9}}' | head -30", timeout=20)
    if r2 == 0 and files:
        s.files = files
    return s


# ---------------------------------------------------------------- 官方模型卡

def fetch_hf_quant_config(model_id: str, proxy: str = "") -> dict[str, Any]:
    """拉取 HF 模型卡的 config（含 quantization_config）。走本机代理。"""
    env = dict(os.environ)
    if proxy:
        env["http_proxy"] = env["HTTP_PROXY"] = proxy
        env["https_proxy"] = env["HTTPS_PROXY"] = proxy
    argv = ["curl", "-sL", "--max-time", "30",
            f"https://huggingface.co/api/models/{model_id}"]
    for k in _HF_PROXY_ENV_KEYS:
        if not proxy:
            env.pop(k, None)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=40, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"拉取 {model_id} 超时（40s）") from None
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"拉取 {model_id} 失败: HTTP {proc.returncode} / {proc.stderr.strip()[:120]}")
    try:
        data = json.loads(proc.stdout)
    except ValueError as ex:      # HF 不可达时代理可能回一段 HTML，别把 traceback 抛给 UI
        raise RuntimeError(f"{model_id} 返回的不是 JSON（代理或网络异常）: {ex}") from ex
    if "config" not in data:
        raise RuntimeError(f"{model_id} 的 config 里没有 quantization_config（可能不是量化模型卡）")
    return data["config"]


def fetch_official_summary(model_id: str, proxy: str = "") -> QuantSummary:
    cfg = fetch_hf_quant_config(model_id, proxy)
    return summarize(cfg.get("quantization_config", {}), source=model_id)


# ---------------------------------------------------------------- 对照

def diff_summaries(mine: QuantSummary, official: QuantSummary) -> list[str]:
    """两条摘要的差异提示（只比位数分布与数量级，target 顺序不同不算差异）。"""
    if mine.error or official.error:
        return ["无法对照：一方数据缺失"]
    if not mine.groups or not official.groups:
        return ["无法对照：缺少 config_groups"]

    def sig(s: QuantSummary) -> dict[str, int]:
        d: dict[str, int] = {}
        for g in s.groups:
            d[str(g["bits"])] = d.get(str(g["bits"]), 0) + g["count"]
        return d

    sm, so = sig(mine), sig(official)
    lines: list[str] = []
    all_bits = sorted(set(sm) | set(so))
    for b in all_bits:
        a, c = sm.get(b, 0), so.get(b, 0)
        mark = "✓" if a == c else "≈" if a and c else "✗"
        lines.append(f"{mark} {b}-bit: 本地 {a} 个 target vs 官方 {c} 个")
    mi, oi = len(mine.ignore), len(official.ignore)
    lines.append(("✓" if mi == oi else "≈") + f" ignore 规则: 本地 {mi} 条 vs 官方 {oi} 条")
    return lines


def to_table_rows(s: QuantSummary) -> list[list[str]]:
    rows = []
    for g in s.groups:
        samples = ", ".join(g["samples"])
        rows.append([g["name"], f"{g['bits']}-bit", str(g["count"]), samples])
    return rows


def ignore_preview(s: QuantSummary, limit: int = 8) -> str:
    if not s.ignore:
        return "（无 ignore 规则）"
    shown = ", ".join(s.ignore[:limit])
    return shown + (f" …共 {len(s.ignore)} 条" if len(s.ignore) > limit else "")

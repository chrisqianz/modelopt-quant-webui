"""显存估算器（设计稿 §6 修订版）。

给定「模型 + 量化格式 + ctx + batch + KV 格式」，估算推理/评测时占多少显存，
并与真实显存总量、当前占用对比，给出「放得下 / 还差 X GB」。

权重量两条路：本机 HF 缓存命中 → `du`；未缓存 → HF API `size` 字段（走代理，
超时 5s）；都失败 → weight_bytes=0, source="unknown"。

KV 两条路：缓存内 config.json 读到层数/KV 头数/头维 → 精确折算；读不到 →
退回经验公式 `weight_bytes * 0.15 * ctx/2048 * batch`，并在 source 追加
`kv_estimate` 标注「估算」。

所有系数（量化折算 / KV 字节）是文件头常量表，可改 —— 估算是辅助判断，
不是精确预算。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from webui import runner
from webui.config import AppConfig

# ---------------------------------------------------------------- 系数表（可改）

# 量化后权重 / 原始权重 的折算系数（NVFP4 固定折算，不做位宽按钮）。
def _quant_factor(qformat: str) -> float:
    q = (qformat or "").lower()
    if q.startswith(("nvfp4", "fp4")):
        return 0.55
    if q.startswith("int4"):
        return 0.45
    if q.startswith("fp8"):
        return 1.0
    return 1.0


# KV 每元素字节数。
def _kv_elem_bytes(kv_qformat: str) -> float:
    q = (kv_qformat or "").lower()
    if q.startswith("fp8"):
        return 1.0
    if q.startswith("nvfp4"):
        return 0.5
    return 0.0


def _hf_dirname(model: str) -> str:
    """org/name → models--org--name（HF 缓存目录命名）。"""
    return "models--" + model.replace("/", "--", 1)


def _weight_bytes(model: str, cfg: AppConfig) -> tuple[int, str]:
    """→ (bytes, source)。source ∈ cache / hf_api / unknown。"""
    hub = cfg.local.hf_cache_path / "hub"
    if not model.startswith(("~", "/")):
        cached = hub / _hf_dirname(model)
        if cached.exists():
            b = _du_bytes(cached)
            if b > 0:
                return b, "cache"
        b, ok = _hf_api_size(model, cfg)
        if ok:
            return b, "hf_api"
        return 0, "unknown"
    # 本机路径：直接 du
    p = Path(model).expanduser()
    b = _du_bytes(p) if p.exists() else 0
    return b, ("cache" if b > 0 else "unknown")


def _du_bytes(d: Path) -> int:
    """du -sb 取字节数（本地目录直接 Path，q_remote 只管 shell 转义）。"""
    try:
        rc, out, _ = runner.run_shell(f"du -sb {runner.q_remote(str(d))}", timeout=30)
        if rc == 0 and out:
            return int(out.split()[0])
    except ValueError:
        pass
    return 0


def _hf_api_size(model: str, cfg: AppConfig) -> tuple[int, bool]:
    """未缓存时问 HF API 的 size 字段（走代理，超时 5s）。"""
    try:
        url = f"https://huggingface.co/api/models/{model}"
        proxy, _ = runner.proxy_url_for(cfg)
        handlers: list[urllib.request.BaseHandler] = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
        opener = urllib.request.build_opener(*handlers)
        with opener.open(url, timeout=5) as r:
            data = json.loads(r.read())
        return int(data.get("size", 0)) or 0, bool(data.get("size"))
    except Exception:
        return 0, False


def _kv_from_config(model: str, cfg: AppConfig) -> dict | None:
    """缓存内 snapshots/*/config.json → {layers, kv_heads, head_dim}；取不到 None。"""
    hub = cfg.local.hf_cache_path / "hub"
    snap_root = hub / _hf_dirname(model) / "snapshots"
    if not snap_root.exists():
        return None
    for snap in snap_root.iterdir():
        cfgj = snap / "config.json"
        if not cfgj.exists():
            continue
        try:
            data = json.loads(cfgj.read_text())
        except (OSError, ValueError):
            continue
        layers = data.get("num_hidden_layers")
        kv_heads = data.get("num_key_value_heads") or data.get("num_attention_heads")
        if not layers or not kv_heads:
            continue
        head_dim = data.get("head_dim")
        if not head_dim:
            hidden = data.get("hidden_size")
            n_heads = data.get("num_attention_heads")
            if hidden and n_heads:
                head_dim = hidden // n_heads
        if not head_dim:
            continue
        return {"layers": layers, "kv_heads": kv_heads, "head_dim": head_dim}
    return None


def _vram_split(cfg: AppConfig) -> tuple[int, int]:
    """probe_host 的 gpu_mem「used/total MiB」→ (used_mib, total_mib)。"""
    st = runner.probe_host(cfg)
    try:
        used_s, total_s = st.gpu_mem.split("/")
        return int(used_s.strip().split()[0]), int(total_s.strip().split()[0])
    except (ValueError, IndexError, AttributeError):
        return 0, 0


def estimate(
    model: str,
    qformat: str,
    kv_qformat: str,
    ctx: int,
    batch: int,
    cfg: AppConfig,
) -> dict:
    """主入口：估算并对比真实显存，返回设计稿 §6 规定的 11 字段 dict。"""
    ctx = max(1, ctx)
    batch = max(1, batch)

    weight_bytes, source = _weight_bytes(model, cfg)
    quant_bytes = int(weight_bytes * _quant_factor(qformat))

    note = ""
    cfg_kv = _kv_from_config(model, cfg)
    elem = _kv_elem_bytes(kv_qformat)
    if elem > 0:
        if cfg_kv is not None:
            kv_bytes = int(
                2 * cfg_kv["layers"] * cfg_kv["kv_heads"] * cfg_kv["head_dim"]
                * ctx * batch * elem
            )
        elif weight_bytes > 0:
            kv_bytes = int(weight_bytes * 0.15 * (ctx / 2048) * batch)
            source += "+kv_estimate"
            note = "KV 走经验公式估算（缓存无 config.json）"
        else:
            kv_bytes = 0
            note = "权重未知且缓存无 config.json，KV 无法估算"
    else:
        kv_bytes = 0

    used_mib, total_mib = _vram_split(cfg)
    need_gib = (quant_bytes + kv_bytes) / (1024 ** 3)
    free_mib = max(0, total_mib - used_mib)
    need_mib = (quant_bytes + kv_bytes) / (1024 ** 2)
    fits = (free_mib - need_mib) > 0
    short_gib = round(need_mib / 1024 - free_mib / 1024, 2) if not fits else 0.0

    return {
        "weight_bytes": weight_bytes,
        "source": source,
        "quant_bytes": quant_bytes,
        "kv_bytes": kv_bytes,
        "vram_total_mib": total_mib,
        "vram_used_mib": used_mib,
        "need_gib": round(need_gib, 2),
        "fits": fits,
        "short_gib": short_gib,
        "note": note,
    }

"""词表：校准数据集 / 官方组合 / qformat / KV qformat。

静态快照做兜底（离线可用、标注 N 项）；运行时从本机仓库目录动态扫描，
ModelOpt 升级后 UI 自动出现新选项，无需改本文件。
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .config import resolve_relative
from .runner import run_shell


def _repo_base(repo: str | Path) -> Path:
    """统一把仓库路径展开（~ / 相对）成绝对路径，再交给 shell 命令。"""
    return resolve_relative(repo)

# ---------------------------------------------------------------- 数据集快照

# 快照来自 modelopt/torch/utils/dataset_utils.py（SUPPORTED_DATASET_CONFIG）。
# key = --dataset 的取值；desc = 界面说明。
DATASETS: list[dict[str, Any]] = [
    {"key": "nemotron-sft-instruction-following-chat-v2", "desc": "Nemotron 指令跟随（对话）"},
    {"key": "nemotron-science-v1", "desc": "Nemotron 科学问答"},
    {"key": "nemotron-competitive-programming-v1", "desc": "Nemotron 竞赛编程"},
    {"key": "nemotron-sft-agentic-v2", "desc": "Nemotron Agentic（工具/代理）"},
    {"key": "nemotron-math-v2", "desc": "Nemotron 数学推理"},
    {"key": "nemotron-sft-swe-v2", "desc": "Nemotron SWE（软件工程）"},
    {"key": "nemotron-sft-multilingual-v1", "desc": "Nemotron 多语言"},
    {"key": "nemotron-post-training-dataset-v2", "desc": "Nemotron 后训练混合 v2", "gated": True},
    {"key": "nemotron-post-training-dataset-v1", "desc": "Nemotron 后训练混合 v1"},
    {"key": "llama-nemotron-post-training-dataset", "desc": "LLaMA 风格 Nemotron 后训练混合"},
    {"key": "open_code_reasoning", "desc": "开源代码推理"},
    {"key": "open_math_reasoning", "desc": "开源数学推理"},
    {"key": "magpie", "desc": "Magpie 通用指令"},
    {"key": "cnn_dailymail", "desc": "CNN/DailyMail 新闻摘要"},
    {"key": "pile", "desc": "The Pile 通用文本"},
    {"key": "pg19", "desc": "PG19 小说"},
    {"key": "wikipedia", "desc": "Wikipedia"},
    {"key": "c4", "desc": "C4 通用网络文本"},
]

# 官方组合（DATASET_COMBOS），一键勾选。
COMBOS: dict[str, list[str]] = {
    "cnn_nemotron_v2_mix": ["cnn_dailymail", "nemotron-post-training-dataset-v2"],
    "nemotron-post-training-v3": [
        "nemotron-sft-instruction-following-chat-v2",
        "nemotron-science-v1",
        "nemotron-competitive-programming-v1",
        "nemotron-sft-agentic-v2",
        "nemotron-math-v2",
        "nemotron-sft-swe-v2",
        "nemotron-sft-multilingual-v1",
    ],
}

COMBO_DESC = {
    "cnn_nemotron_v2_mix": "官方缺省组合（2 数据集）",
    "nemotron-post-training-v3": "官方 Nemotron 后训练组合（7 数据集，NVFP4 常用）",
}


def builtin_dataset_keys() -> list[str]:
    return [d["key"] for d in DATASETS]


def dataset_label(key: str) -> str:
    for d in DATASETS:
        if d["key"] == key:
            return f"{d['key']} — {d['desc']}"
    if key in COMBOS:
        return f"{key}（组合，{len(COMBOS[key])} 数据集）"
    return key  # 自定义 HF 名 / 本地路径


# ---------------------------------------------------------------- qformat 快照

# 快照来自 modelopt_recipes/configs/ptq/presets/model/*.yaml（--qformat 词表）。
QGroup = dict[str, Any]        # {"name": str, "items": list[str]}

QFORMAT_GROUPS: list[QGroup] = [
    {"name": "NVFP4 默认（W4A4）", "items": ["nvfp4", "nvfp4_fp8_mha", "nvfp4_four_over_six"]},
    {"name": "NVFP4 权重范围（官方推荐档）",
     "items": ["nvfp4_mlp_only", "nvfp4_experts_only", "nvfp4_omlp_only", "nvfp4_mlp_weight_only"]},
    {"name": "NVFP4 混合精度",
     "items": ["w4a16_nvfp4", "w4a8_nvfp4_fp8", "w4a8_mxfp4_fp8", "w4a8_awq_beta"]},
    {"name": "NVFP4 校准算法增强",
     "items": ["nvfp4_awq_lite", "nvfp4_awq_clip", "nvfp4_awq_full",
               "nvfp4_w4a4_weight_mse_fp8_sweep", "nvfp4_w4a4_weight_local_hessian",
               "nvfp4_svdquant"]},
    {"name": "FP8", "items": ["fp8", "fp8_2d_blockwise_weight_only", "fp8_per_channel_per_token"]},
    {"name": "INT8", "items": ["int8", "int8_weight_only", "int8_smoothquant"]},
    {"name": "INT4", "items": ["int4_awq", "int4_blockwise_weight_only"]},
    {"name": "MX 家族", "items": ["mxfp4", "mxfp4_mlp_weight_only", "mxfp6", "mxfp8", "mxint8"]},
    {"name": "IQ 家族", "items": ["iq1_s", "iq1_m", "iq2_s", "iq2_xs", "iq2_xxs", "q8_0"]},
    {"name": "Mamba-MoE",
     "items": ["mamba_moe_nvfp4_aggressive", "mamba_moe_nvfp4_conservative",
               "mamba_moe_fp8_aggressive", "mamba_moe_fp8_conservative"]},
]

KV_QFORMATS: list[str] = ["none", "fp8", "fp8_affine", "fp8_cast",
                          "nvfp4", "nvfp4_affine", "nvfp4_cast", "nvfp4_rotate"]


def all_qformats() -> list[str]:
    return [i for g in QFORMAT_GROUPS for i in g["items"]]


def kv_choices(kv: list[str] | None = None) -> list[tuple[str, str]]:
    """kv 下拉 choices：(展示名, 实际值) 元组，与 qf_dd 形态一致。
    保持纯字符串 choices 会在刷新前后形态不一，容易让前端把 (name, value)
    对整体发回（"value not in choices" 报错）。"""
    return [(("none（不量化 KV）" if k == "none" else k), k) for k in (kv or KV_QFORMATS)]


# ---------------------------------------------------------------- 仓库动态扫描

def scan_qformats(repo: str | Path) -> dict[str, object]:
    """扫描仓库预设目录里的 qformat / kv 词表。

    Returns:
        {"model": [names...], "kv": [names...], "source": "remote|snapshot", "error": ""}
        source 仍是 "remote"/"snapshot" 两个旧取值，UI 靠它判断要不要提示。
    """
    sub = {"model": "modelopt_recipes/configs/ptq/presets/model",
           "kv": "modelopt_recipes/configs/ptq/presets/kv"}
    out: dict[str, object] = {"source": "remote", "error": ""}
    try:
        for kind, d in sub.items():
            # 必须是绝对路径：run_shell 不 cd 进仓库，给相对路径会在 $HOME 里执行
            # （这就是旧版「刷新词表」从来没生效过的原因）
            abs_d = shlex.quote(str(_repo_base(repo) / d))
            r, o, e = run_shell(
                f"ls {abs_d} 2>/dev/null | grep -E '\\.(yaml|yml)$' | sed 's/\\..*//' | sort",
                timeout=20,
            )
            if r != 0 or not o.strip():
                raise RuntimeError(f"仓库预设目录不可读: {d} ({e or o})")
            out[kind] = o.split()
        return out
    except Exception as ex:  # noqa: BLE001 - 快照兜底
        return {"model": all_qformats(), "kv": list(KV_QFORMATS),
                "source": "snapshot", "error": str(ex)}


def scan_datasets(repo: str | Path) -> dict[str, object]:
    """从仓库的 dataset_utils.py 解析数据集注册表与组合（纯文本解析，不 import modelopt）。"""
    src_path = shlex.quote(str(_repo_base(repo) / "modelopt/torch/utils/dataset_utils.py"))
    r, src, e = run_shell(f"cat {src_path} 2>/dev/null", timeout=20)
    if r != 0 or "SUPPORTED_DATASET_CONFIG" not in src:
        return {"datasets": builtin_dataset_keys(), "combos": dict(COMBOS),
                "source": "snapshot", "error": e or "解析失败"}
    i = src.find("SUPPORTED_DATASET_CONFIG")
    j = src.find("}", i)
    block = src[i:j if j > 0 else i + 8000]
    keys = re.findall(r'^\s{4}"([^"]+)"\s*:', block, re.M)
    i = src.find("DATASET_COMBOS")
    j = src.find("}", src.find("]", i))
    combos_block = src[i:j if j > 0 else i + 2000]
    combos = dict(re.findall(r'(?m)^\s{4}"([^"]+)":\s*\[(.*?)\]', combos_block))
    combos = {k: re.findall(r'"([^"]+)"', v) for k, v in combos.items()}
    if not keys:
        return {"datasets": builtin_dataset_keys(), "combos": dict(COMBOS),
                "source": "snapshot", "error": "解析失败"}
    return {"datasets": keys, "combos": combos, "source": "remote", "error": ""}


def render_qformat_dropdown(options: list[str], groups: list[QGroup] | None = None) -> list[tuple[str, str]]:
    """把 qformat 列表转成 Gradio choices [(展示名, 实际值), ...]，带分组注释。"""
    choices: list[tuple[str, str]] = []
    if groups is None:
        groups = QFORMAT_GROUPS
    seen: set[str] = set()
    for g in groups:
        for item in g["items"]:
            if item in options and item not in seen:
                choices.append((f"[{g['name']}] {item}", item))
                seen.add(item)
    # 远程刷新后可能出现的新 preset（快照分组里没列到的）
    for item in options:
        if item not in seen:
            choices.append((f"[远程新增] {item}", item))
    return choices

# ---------------------------------------------------------------- recipe 菜单

# 组顺序即菜单分组顺序；(仓库内子目录, 组名)。configs/ 是被 imports 引用的
# 组件（numerics/units），不能独立传给 --recipe，故意不扫。
_RECIPE_GROUPS: tuple[tuple[str, str], ...] = (
    ("general/ptq", "官方通用"),
    ("model_type", "按模型架构"),
    ("models", "模型专属"),
)


def scan_recipes(repo: str | Path) -> dict[str, object]:
    """运行时扫仓库 modelopt_recipes/，产出 --recipe 可直接填的菜单。

    判据与 modelopt/recipe/loader.py 对齐，两种合法形态都认：
    ① dict 形式：任一文档顶层有 ``quantize`` 段；② quant_cfg list 形式：
    任一文档是非空 dict 列表（仓库里有双文档写法：doc1=imports、
    doc2=quantizer 列表，safe_load 会报多文档错，故用 safe_load_all）。
    name = 相对 modelopt_recipes/ 的路径去掉后缀，就是 loader 认的写法。

    Returns:
        {"recipes": [{"name","group","description"}...], "error": "",
         "skipped": ["坏文件的解析错误，供排查"]}
        仓库不可读时 recipes=[] + error，前端自动退回纯文本输入。
    """
    import yaml  # noqa: PLC0415 - 只这里需要，别拖慢模块导入

    base = _repo_base(repo) / "modelopt_recipes"
    try:
        if not base.is_dir():
            raise RuntimeError(f"recipe 库目录不存在: {base}")
        recipes: list[dict[str, str]] = []
        skipped: list[str] = []
        for sub, group in _RECIPE_GROUPS:
            root = base / sub
            if not root.is_dir():
                continue
            for f in sorted(root.rglob("*")):
                if not f.is_file() or f.suffix not in (".yaml", ".yml"):
                    continue
                try:
                    docs = [d for d in yaml.safe_load_all(f.read_text()) if d]
                except Exception as ex:  # noqa: BLE001 - 单文件坏了不拖垮整表
                    skipped.append(f"{f.relative_to(base)}: {ex.__class__.__name__}: {ex}")
                    continue
                is_recipe = any(isinstance(d, dict) and "quantize" in d for d in docs) or any(
                    isinstance(d, list) and d and all(isinstance(e, dict) for e in d)
                    for d in docs)
                if not is_recipe:
                    continue
                meta = next((d.get("metadata") for d in docs
                             if isinstance(d, dict) and isinstance(d.get("metadata"), dict)),
                            None)
                desc = " ".join(str((meta or {}).get("description") or "").split())
                recipes.append({"name": str(f.relative_to(base).with_suffix("")),
                                "group": group, "description": desc[:200]})
        # 去重：仓库惯例 xxx.yaml 是完整 recipe、xxx.quant_cfg.yaml 是它的
        # 内层列表件；两者同现只留 xxx。孤儿 .quant_cfg（如 qwen3_vl 的
        # vision_fp8）是唯一入口，保留。
        names = {r["name"] for r in recipes}
        recipes = [r for r in recipes
                   if not (r["name"].endswith(".quant_cfg")
                           and r["name"][:-len(".quant_cfg")] in names)]
        return {"recipes": recipes, "error": "", "skipped": skipped}
    except Exception as ex:  # noqa: BLE001 - 降级为自由输入，不阻塞页面
        return {"recipes": [], "error": str(ex), "skipped": []}


def check_recipe(repo: str | Path, recipe: str) -> str | None:
    """发射前校验 --recipe 引用。返回 None=可解析；否则给人看的错误语。

    解析顺序对齐 modelopt/recipe/loader.py::_resolve_recipe_path：hf_ptq 是
    ``cd 仓库 && python`` 起的，所以先按「相对仓库」探测，再退到
    modelopt_recipes/ 下找，各探测 原样/.yaml/.yml（文件或目录）；绝对路径
    直接探测。仓库目录不可读时放行——与 capabilities 探测失败不拦截同理，
    宁可漏拦不误杀。
    """
    from contextlib import suppress

    recipe = (recipe or "").strip()
    if not recipe:
        return None
    base = _repo_base(repo)
    if not base.is_dir():
        return None

    def _hit(p: Path) -> bool:
        if p.is_file() or p.is_dir():
            return True
        with suppress(ValueError):  # 无 name / 怪后缀时 with_suffix 会抛
            for suf in (".yaml", ".yml"):
                if p.with_suffix(suf).is_file():
                    return True
        return False

    cand = Path(recipe).expanduser()
    if cand.is_absolute():
        return None if _hit(cand) else f"recipe 文件不存在: {recipe}"
    if _hit(base / recipe) or _hit(base / "modelopt_recipes" / recipe):
        return None
    return (f"recipe 不存在: {recipe} —— 从菜单里选，"
            f"或填 GPU 机（{base.parent}）上的 YAML 绝对路径")

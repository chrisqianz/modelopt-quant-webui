"""量化命令拼装与结果解析（不依赖 UI 框架）。

从 app.py 顶部搬出：app.py（Gradio 入口）与未来的 server/（FastAPI 入口）
从这里 import。函数一律接受 cfg: AppConfig 参数，不读任何模块级配置，
两个入口可以带各自的配置进来。量化命令串与搬前逐字节一致（tests/golden 钉死）。
"""

from __future__ import annotations

import re
import shlex

from webui import runner
from webui.config import AppConfig
from webui.runner import q_remote

# ---------------------------------------------------------------- 评测词表

# lm-eval-harness 任务名（单卡可跑、无需判分端点）。
# AA 套件（HLE/SciCode/AA-LCR/Omniscience/τ²/Terminal-Bench）需要判分端点 +
# 沙箱/长上下文，单卡裸机跑不动——见评测 tab 的说明与 NEL 引导。
EVAL_PRESETS: dict[str, list[str]] = {
    "AA 标准（GPQA + MMLU-Pro + IFBench）":
        ["gpqa_diamond_cot_zeroshot", "mmlu_pro", "ifeval"],
    "快速检查（GPQA 单项）": ["gpqa_diamond_cot_zeroshot"],
    "量化敏感扩展（+ 数学/知识/代码/指令）":
        ["gpqa_diamond_cot_zeroshot", "mmlu_pro", "ifeval",
         "gsm8k_cot_zeroshot", "hellaswag", "humaneval_instruct"],
}

EVAL_TASKS = [
    "gpqa_diamond_cot_zeroshot", "gpqa_diamond_cot_n_shot", "gpqa_main_cot_zeroshot",
    "mmlu_pro", "mmlu", "arc_challenge", "hellaswag", "gsm8k_cot_zeroshot",
    "ifeval", "humaneval_instruct", "drop",
]


def fmt_val(x) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


# ---------------------------------------------------------------- 命令拼装

def build_params(a: list, cfg: AppConfig) -> dict:
    """把事件回调位置参数解析成 build_cmd 需要的 dict。
    a[0..19] 对应 UI 参数区固定顺序（见 INPUT_ORDER）。"""
    sizes: dict[str, int] = {}
    for i, name in enumerate(a[6] or []):  # sizes_names 勾选顺序
        row = a[7][i] if a[7] and i < len(a[7]) else None
        if row and row[0] is not None:
            sizes[name] = int(row[0])
    # 前端有时会把 (展示名, 值) 对整体发回（多一层 list），取最后一项兜底
    qf = a[2]
    if isinstance(qf, (list, tuple)) and qf:
        qf = qf[-1]
    kv = a[4]
    if isinstance(kv, (list, tuple)) and kv:
        kv = kv[-1]
    return {
        "model": a[0], "trust_remote_code": a[1], "qformat": qf,
        "recipe": a[3], "kv_cache_qformat": kv, "datasets": a[5],
        "dataset_sizes": sizes, "custom_datasets": a[8],
        "calib_seq": a[9], "calib_with_images": a[10], "export_path": a[11],
        "batch_size": a[12], "device": a[13], "attn_implementation": a[14],
        "use_fsdp2": a[15], "cpu_offload": a[16], "low_memory_mode": a[17],
        "gpu_max_mem_percentage": a[18], "skip_generate": a[19],
    }


def is_local_model(model: str) -> bool:
    """模型填的是路径而不是 HF ID（~ 或 / 开头）。"""
    m = (model or "").strip()
    return m.startswith(("~", "/"))


def task_slug(model: str) -> str:
    """任务目录名：本地路径取末段目录名，HF ID 保留 org__model。
    不然 /data/models/x 会变成 __data__models__x_时间戳 这种没法认的目录。"""
    m = (model or "").strip().rstrip("/")
    if is_local_model(m):
        m = m.rsplit("/", 1)[-1] or "model"
    else:
        m = m.replace("/", "__")
    return re.sub(r"[^\w.\-]+", "_", m)[:60] or "model"


def check_local_model(cfg: AppConfig, model: str) -> str:
    """本地模型开跑前在本机确认一下，别等几分钟后才告诉你路径写错了。
    返回 "" = 没问题，否则是给用户看的错误说明。"""
    probe = (f"P={runner.q_remote(model)}; "
             'if [ -d "$P" ]; then '
             'if [ -f "$P/config.json" ]; then echo __OK__; else echo __NOCFG__; fi; '
             "else echo __NODIR__; fi")
    _, flag, _ = runner.run_shell(probe, timeout=30)
    flag = flag or ""
    if "__OK__" in flag:
        return ""
    if "__NODIR__" in flag:
        return (f"没有找到这个目录：{model}\n"
                "路径写在 WebUI 所在这台机器上。")
    return f"{model} 在，但里面没有 config.json，不像 HF 模型目录。"


def build_cmd(p: dict, cfg: AppConfig) -> str:
    """把 UI 参数拼成本机 hf_ptq.py 命令（含 cd 与 venv python）。
    会被包进 nohup bash -c '...'，所以 ~ 前缀路径一律走 q_remote 转绝对路径。"""
    repo = q_remote(cfg.local.repo)
    venv = q_remote(cfg.local.venv)
    parts = [f"cd {repo} && {venv}/bin/python examples/hf_ptq/hf_ptq.py"]

    model = (p.get("model") or "").strip()
    if not model:
        raise ValueError("请填写模型路径/名称")
    parts.append(f"--pyt_ckpt_path "
                 f"{q_remote(model) if is_local_model(model) else shlex.quote(model)}")
    if p.get("trust_remote_code"):
        parts.append("--trust_remote_code")

    qfmt = (p.get("qformat") or "").strip()
    recipe = (p.get("recipe") or "").strip()
    if recipe:
        parts.append(f"--recipe {q_remote(recipe) if recipe.startswith(('/', '~')) else shlex.quote(recipe)}")
    elif qfmt:
        parts.append(f"--qformat {shlex.quote(qfmt)}")
    kv = (p.get("kv_cache_qformat") or "none")
    if kv and kv != "none":
        parts.append(f"--kv_cache_qformat {shlex.quote(kv)}")

    # 校准数据集：内置勾选 + 自定义行，逐行独立 calib_size
    ds_names: list[str] = []
    ds_sizes: list[str] = []
    sizes = p.get("dataset_sizes") or {}
    default_size = str(int(cfg.defaults.get("calib_size") or 1024))
    for name in p.get("datasets") or []:
        ds_names.append(name)
        ds_sizes.append(str(int(sizes.get(name) or default_size)))
    for row in p.get("custom_datasets") or []:
        # 单列 DataFrame 会把整表折成「每行一个标量」（['tiny']），
        # 这里补回一行两列的形状，否则用户填了但命令里没有且无提示。
        if isinstance(row, (str, int, float)):
            row = [row]
        if not isinstance(row, (list, tuple)) or len(row) < 1:
            continue
        name = str(row[0]).strip()
        if not name:
            continue
        size = row[1] if len(row) > 1 and row[1] is not None else None
        ds_names.append(name)
        ds_sizes.append(str(int(size)) if size else default_size)
    if ds_names:
        parts.append(f"--dataset {shlex.quote(','.join(ds_names))}")
        parts.append(f"--calib_size {shlex.quote(','.join(ds_sizes))}")

    parts.append(f"--calib_seq {int(p.get('calib_seq') or cfg.defaults.get('calib_seq') or 512)}")
    if p.get("calib_with_images"):
        parts.append("--calib_with_images")

    export = (p.get("export_path") or "").strip()
    if export:
        parts.append(f"--export_path {q_remote(export) if export.startswith(('/', '~')) else shlex.quote(export)}")

    parts.append(f"--batch_size {int(p.get('batch_size') or 0)}")
    parts.append(f"--device {shlex.quote(p.get('device') or 'cuda')}")
    attn = (p.get("attn_implementation") or "").strip()
    if attn:
        parts.append(f"--attn_implementation {shlex.quote(attn)}")
    if p.get("use_fsdp2"):
        parts.append("--use_fsdp2")
    if p.get("cpu_offload"):
        parts.append("--cpu_offload")
    if p.get("low_memory_mode"):
        parts.append("--low_memory_mode")
    gmp = p.get("gpu_max_mem_percentage")
    if gmp:
        parts.append(f"--gpu_max_mem_percentage {float(gmp)}")
    if p.get("skip_generate"):
        parts.append("--skip_generate")

    return " ".join(parts)


# ---------------------------------------------------------------- 评测脚本

def build_eval_script(quant: str, baseline: str,
                      tasks: list[str], tp: int, port: int,
                      batch: str, cfg: AppConfig) -> str:
    """拼评测流水线脚本：读量化元数据选 flag → vLLM 部署量化 checkpoint →
    跑 lm_eval →（可选）部署基线 → 再跑 → 结果分落 quant/ 与 baseline/ 目录。
    脚本先落盘再执行（start_script_task），避免多层 bash -c 引号嵌套。"""
    q = quant.strip()
    b = (baseline or "").strip()
    if not q:
        raise ValueError("请填量化模型路径")
    t = ",".join(x.strip() for x in tasks if x and x.strip())
    if not t:
        raise ValueError("请选评测任务")
    tp = max(1, int(tp or 1))
    port = int(port or 8000)
    batch = str(batch or "auto").strip() or "auto"
    qq = shlex.quote(q)
    bb = shlex.quote(b)
    ma = f"model=served,base_url=http://127.0.0.1:{port}/v1/completions,tokenized_requests=False"

    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        'OUTDIR="$(pwd)"',
        f'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 P="{port}" TP="{tp}"',
        "# ---- 0. 从 checkpoint 元数据推断 vLLM quantization flag ----",
        f'python - {qq} <<\'__MO_PY__\' > /tmp/.mo_algopair 2>/dev/null || true',
        "import json, os, sys",
        "try:",
        '    d = json.load(open(os.path.join(sys.argv[1], "hf_quant_config.json")))',
        "except Exception:",
        "    d = {}",
        'print(d.get("quantization_algo", ""), d.get("quant_method", ""))',
        "__MO_PY__",
        'ALGO_PAIR=$(cat /tmp/.mo_algopair 2>/dev/null)',
        'read -r ALGO _ <<< "$ALGO_PAIR"',
        'case "$ALGO" in',
        "  *NVFP4*|*nvfp4*|*FP4*|*fp4*) QF=modelopt_fp4;;",
        '  "") echo "[webui] ⚠ 无 hf_quant_config.json，按未量化加载"; QF="";;',
        "  *) QF=modelopt;;",
        "esac",
        'echo "[webui] quantization flag = $QF (algo=$ALGO)"',
        "serve(){",
        '  local m="$1" f="$2"',
        '  pkill -f "vllm serve" 2>/dev/null; sleep 3',
        '  echo "[webui] serve $m flag=$f port=$P tp=$TP"',
        '  if [ -n "$f" ]; then',
        '    vllm serve "$m" --quantization "$f" --tensor-parallel-size $TP \\',
        '    --host 127.0.0.1 --port $P --served-model-name served >/tmp/vllm.log 2>&1 &',
        "  else",
        '    vllm serve "$m" --tensor-parallel-size $TP \\',
        '    --host 127.0.0.1 --port $P --served-model-name served >/tmp/vllm.log 2>&1 &',
        "  fi",
        "  echo $! > /tmp/vllm.pid",
        "  ok=0",
        "  for i in $(seq 1 240); do",
        "    sleep 5",
        '    if curl -sf "http://127.0.0.1:$P/health" >/dev/null 2>&1; then ok=1; break; fi',
        '    if ! kill -0 $(cat /tmp/vllm.pid) 2>/dev/null; then break; fi',
        "  done",
        '  if [ $ok -ne 1 ]; then echo "[webui] vLLM 启动失败，日志:"; tail -n 40 /tmp/vllm.log; exit 1; fi',
        "  echo '[webui] server healthy'",
        "}",
        "evalm(){",
        '  local out="$1"',
        '  mkdir -p "$out"',
        '  echo "[webui] lm_eval tasks=$t out=$out"',
        '  lm_eval --model local-completions \\',
        f'    --model_args "{ma}" \\',
        f'    --batch_size "{batch}" --tasks "$t" --output_path "$out" \\',
        '  >> "$OUTDIR/lmeval.log" 2>&1 || echo "[webui] ⚠ lm_eval 失败，见 lmeval.log"',
        "}",
        'cleanup(){ pkill -f "vllm serve" 2>/dev/null; echo "[webui] vllm stopped"; }',
        "trap cleanup EXIT",
        "# ---- 1. 量化模型 ----",
        f'serve {qq} "$QF"',
        'evalm "$OUTDIR/quant"',
    ]
    if b:
        lines += [
            "# ---- 2. 基线模型（BF16）----",
            f'serve {bb} ""',
            'evalm "$OUTDIR/baseline"',
        ]
    else:
        lines += ["# ---- 2. 基线：跳过（未填基线模型）----"]
    lines += [
        "# ---- 3. 结果 ----",
        "echo '[webui] 结果文件:'",
        'ls -1t "$OUTDIR"/quant/results_*.json "$OUTDIR"/baseline/results_*.json 2>/dev/null',
        "echo EVAL_DONE",
    ]
    return "\n".join(lines)


def result_tables(results: list[dict]) -> tuple[list[list[str]], str]:
    """把多个 lm_eval results json 解析成对比表。
    results[0] = 量化，results[1] = 基线（fetch_eval_results 保证顺序）。
    返回 (rows, note)。rows: [模型, 任务, 指标, 量化, 基线, Δ(量化-基线)]。"""
    rows: list[list[str]] = []
    if not results:
        return rows, "（无结果文件）"
    q = results[0].get("data") or {}
    base = results[1].get("data") or {} if len(results) > 1 else {}

    def task_metric(d, tk):
        """从 results json 取任务的第一个主指标 (key, value)。"""
        rec = (d.get("results") or {}).get(tk) or {}
        cand = []
        for k, v in rec.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and k.split(",")[0] == tk:
                cand.append((k, v))
        if not cand:
            return None, None

        def prio(kv):
            parts = kv[0].split(",")
            m = parts[1] if len(parts) > 1 else ""
            s = parts[2] if len(parts) > 2 else "none"
            rank = 0 if m == "acc" else 1 if "stderr" not in m else 2
            return (rank, 0 if s == "none" else 1)

        cand.sort(key=prio)
        return cand[0]

    tasks = sorted(set(list((q.get("results") or {}).keys())
                       + list((base.get("results") or {}).keys())))
    for tk in tasks:
        km, qv = task_metric(q, tk)
        _, bv = task_metric(base, tk)
        delta = ""
        if isinstance(qv, (int, float)) and isinstance(bv, (int, float)):
            delta = f"{qv - bv:+.4f}"
        if qv is None and bv is None:
            continue
        rows.append(["量化", tk, km or "",
                     fmt_val(qv) if qv is not None else "",
                     fmt_val(bv) if bv is not None else "", delta])
    note = "（仅量化模型，未跑基线）" if not base else ""
    return rows, note

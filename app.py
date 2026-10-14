"""ModelOpt 量化 WebUI 主入口（Gradio）。

布局：状态栏（GPU 探活 / venv / 显存）→ [量化] tab（模型 → 量化配置 →
校准数据集 → 高级参数 → 命令预览 → 启动/日志 → 结果校验）→ [评测] tab
（vLLM 部署 + lm-eval 跑「量化 vs 基线」对比，回答"量化有没有吃瘪"）。

跑在 GPU 机器上，浏览器从局域网打开。路径与代理地址一律在 UI 里填，
代码里不写死。
"""

from __future__ import annotations

import contextlib
import json
import re
import shlex
import time

import gradio as gr
import gradio.themes as gr_themes

from webui import progress, runner, validate, vocab
from webui.config import (
    DEFAULT_HF_CACHE,
    DEFAULT_REPO,
    DEFAULT_TASK_DIR,
    DEFAULT_VENV,
    PROXY_AUTO,
    PROXY_DIRECT,
    PROXY_MANUAL,
    PROXY_MODES,
    NetConfig,
    load_config,
    save_local,
    save_net,
)
from webui.runner import q_remote

CFG = load_config()
DEFAULTS = CFG.defaults

# 参数输入区的固定顺序（事件绑定的唯一事实源，别动顺序）
INPUT_ORDER = [
    "model", "trust_remote_code", "qformat", "recipe", "kv_cache_qformat",
    "datasets", "sizes_names", "sizes_table", "custom_datasets", "calib_seq",
    "calib_with_images", "export_path", "batch_size", "device",
    "attn_implementation", "use_fsdp2", "cpu_offload", "low_memory_mode",
    "gpu_max_mem_percentage", "skip_generate",
]

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

def build_params(a: list) -> dict:
    """把事件回调位置参数解析成 build_cmd 需要的 dict。
    a[0..19] 对应 UI 参数区固定顺序（见 INPUT_ORDER）。"""
    sizes: dict[str, int] = {}
    for i, name in enumerate(a[6] or []):  # sizes_names 勾选顺序
        row = a[7][i] if a[7] and i < len(a[7]) else None
        if row and row[0] is not None:
            sizes[name] = int(row[0])
    # gradio 6 前端有时会把 (展示名, 值) 对整体发回（多一层 list），取最后一项兜底
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


def check_local_model(model: str) -> str:
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


def build_cmd(p: dict) -> str:
    """把 UI 参数拼成本机 hf_ptq.py 命令（含 cd 与 venv python）。
    会被包进 nohup bash -c '...'，所以 ~ 前缀路径一律走 q_remote 转绝对路径。"""
    repo = q_remote(CFG.local.repo)
    venv = q_remote(CFG.local.venv)
    parts = [f"cd {repo} && {venv}/bin/python examples/hf_ptq/hf_ptq.py"]

    model = (p.get("model") or "").strip()
    if not model:
        raise gr.Error("请填写模型路径/名称")
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
    default_size = str(int(DEFAULTS.get("calib_size") or 1024))
    for name in p.get("datasets") or []:
        ds_names.append(name)
        ds_sizes.append(str(int(sizes.get(name) or default_size)))
    for row in p.get("custom_datasets") or []:
        # gradio 单列 DataFrame 会把整表折成「每行一个标量」（['tiny']），
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

    parts.append(f"--calib_seq {int(p.get('calib_seq') or DEFAULTS.get('calib_seq') or 512)}")
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
                      batch: str) -> str:
    """拼评测流水线脚本：读量化元数据选 flag → vLLM 部署量化 checkpoint →
    跑 lm_eval →（可选）部署基线 → 再跑 → 结果分落 quant/ 与 baseline/ 目录。
    脚本先落盘再执行（start_script_task），避免多层 bash -c 引号嵌套。"""
    q = quant.strip()
    b = (baseline or "").strip()
    if not q:
        raise gr.Error("请填量化模型路径")
    t = ",".join(x.strip() for x in tasks if x and x.strip())
    if not t:
        raise gr.Error("请选评测任务")
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


def _result_tables(results: list[dict]) -> tuple[list[list[str]], str]:
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
# ---------------------------------------------------------------- 进度显示

_STAGE_CSS = """
<div style="margin:2px 0 8px;padding:10px 12px;border-radius:10px;
background:rgba(16,185,129,.07);border:1px solid rgba(148,163,184,.25)">
<div style="font-size:.86em;font-weight:600;letter-spacing:.02em">{head}</div>
<div style="margin:7px 0 4px;height:10px;border-radius:999px;
background:rgba(148,163,184,.22);overflow:hidden">{fill}</div>
<div style="font-size:.82em;opacity:.85;font-variant-numeric:tabular-nums">{meta}</div>
</div>
"""


def _prog_html(rep, note: str = "") -> str:
    """ProgressReport → 进度条卡片。gr.HTML 会清掉 script/style，故只用内联样式。"""
    n = len(progress.STAGES)
    dots = "  ".join(
        ("●" if i <= rep.stage_idx else "○") for i in range(n))
    if rep.error:
        head = f"⛔ {rep.stage}"
        fill = ('<div style="height:100%;width:100%;border-radius:999px;'
                'background:#ef4444"></div>')
        meta = rep.error
    elif rep.done:
        head = "✅ 完成"
        fill = ('<div style="height:100%;width:100%;border-radius:999px;'
                'background:#10b981"></div>')
        meta = rep.detail or rep.item
    else:
        head = f"{rep.stage_idx + 1}/{n} · {rep.stage}"
        pct = rep.pct
        if pct is None:
            fill = ('<div style="height:100%;width:35%;border-radius:999px;'
                    'background:linear-gradient(90deg,#0ea5e9,#34d399)"></div>')
        else:
            fill = (f'<div style="height:100%;width:{max(2, min(100, pct)):.1f}%;'
                    'border-radius:999px;background:linear-gradient(90deg,#0ea5e9,#34d399)">'
                    '</div>')
        parts = []
        if pct is not None:
            parts.append(f"{pct:.1f}%")
        if rep.detail:
            parts.append(rep.detail)
        if rep.item:
            parts.append(rep.item)
        meta = "  ·  ".join(parts) or "等待输出…"
    if note:
        meta = f"{meta}  {note}" if meta else note
    return _STAGE_CSS.format(head=f"{head}<div style='opacity:.6;font-weight:400'>"
                                f"{dots}</div>", fill=fill, meta=meta or "—")


def _prog_idle(text: str = "等待任务") -> str:
    return _prog_html(progress.ProgressReport(stage_idx=0, stage=text))


def _prog_text(rep) -> str:
    """进度压成一行纯文本（评测日志用）。"""
    pct = "" if rep.pct is None else f" {rep.pct:.0f}%"
    tail = " · ".join(x for x in (rep.detail, rep.item) if x)
    return f"▸ {rep.stage}{pct}" + (f"  {tail}" if tail else "")


_EV_STAGES = ["准备启动 vllm", "等 server 就绪", "跑 lm_eval 评测", "停止 server", "完成"]


def _ev_progress(log: str):
    """评测脚本的进度：靠 build_eval_script 里 echo 的 [webui] 标记定位阶段。"""
    low = log.lower()
    idx = 0
    if "[webui] lm_eval" in low or "running on" in low or "it/s" in low:
        idx = 2
    elif "server healthy" in low or "启动 server" in low:
        idx = 1
    elif "stopping server" in low or "[webui] vllm" in low:
        idx = 3
    if low.strip().rstrip(":").endswith("eval_done"):
        idx = 4
    rep = progress.parse(log)
    tail = "\n".join(log.splitlines()[-3:]) if log else ""
    stage = "完成" if idx == 4 else _EV_STAGES[idx]
    if idx == 2:                      # 评测中：lm_eval 自带 tqdm，用它的百分比
        return progress.ProgressReport(stage_idx=idx, stage=stage, pct=rep.pct,
                                       detail=rep.detail, item=tail[-90:])
    return progress.ProgressReport(stage_idx=idx, stage=stage, item=tail[-90:])


# ---------------------------------------------------------------- 页面

def _env_line() -> str:
    """顶栏那颗「跑在哪」的胶囊 —— 只有一台机器，就是这台。"""
    return "**执行位置**：这台机器"


def _net_line() -> str:
    """顶栏那颗「HF 下载走哪」的胶囊（不探测，只说当前配置）。"""
    net = CFG.net
    if net.mode == PROXY_DIRECT:
        return "**HF 下载**：直连（不注入代理）"
    url, note = runner.proxy_url_for(CFG)
    use = url or net.proxy
    if net.mode == PROXY_MANUAL:
        return f"**HF 下载**：强制代理 `{use or '⚠ 还没填地址'}`" + (f"（{note}）" if note else "")
    if use:
        return f"**HF 下载**：自动 · 直连优先，不通走 `{use}`"
    return "**HF 下载**：自动 · 直连优先（未填代理地址，直连不通就下不动）"


MO_CSS = """
.mo-side { border-right: 1px solid var(--border-color-primary);
           padding-right: 14px; }
.mo-brand p { font-weight: 700; letter-spacing: -.01em; margin-bottom: 0; }
.mo-sub p { color: var(--body-text-color-subdued); font-size: .8rem; }
.mo-nav .wrap { flex-direction: column; align-items: stretch; gap: 4px; }
.mo-nav label { display: flex; align-items: center; gap: 8px;
                padding: 8px 12px; border-radius: 10px; cursor: pointer;
                border: 1px solid transparent; font-size: .95rem; }
.mo-nav label:hover { background: var(--neutral-100); }
.mo-nav label:has(input:checked) { background: var(--primary-100);
                border-color: var(--primary-300); color: var(--primary-900);
                font-weight: 600; }
.mo-nav input[type=radio] { accent-color: var(--primary-500); }
"""

def _build_app() -> gr.Blocks:
    qf_choices = vocab.render_qformat_dropdown(vocab.all_qformats())
    kv_choices0 = vocab.kv_choices()
    ds_all = [d["key"] for d in vocab.DATASETS]

    with gr.Blocks(title="ModelOpt 量化 WebUI") as app:
        with gr.Row(equal_height=False):
            with gr.Column(scale=0, min_width=252, elem_classes=["mo-side"]):
                gr.Markdown("#### ModelOpt 量化台", elem_classes=["mo-brand"])
                gr.Markdown("HF 权重 → NVFP4 / FP8 / INT4",
                          elem_classes=["mo-sub"])
                nav = gr.Radio(["量化", "评测", "对配方", "连接设置"],
                               value="量化", show_label=False,
                               container=False,
                               elem_classes=["mo-nav"])
                refresh_btn = gr.Button("⟳ 探测", variant="secondary", size="sm")
                mode_md = gr.Markdown(_env_line())
                net_md = gr.Markdown(_net_line())
                venv_btn = gr.Button("一键创建 venv", size="sm")
                eval_env_btn = gr.Button("一键装评测环境", size="sm")
            with gr.Column(scale=1, min_width=0):
                status_md = gr.Markdown("点 **⟳ 探测** 看 GPU / venv / 磁盘 / HF 网络实测情况。")
                venv_log = gr.Markdown("", elem_classes=["mo-log"])

                # ============================================ 量化 =====
                with gr.Column(visible=True, elem_classes=["mo-pane"]) as pane_quant:  # noqa: SIM117
                    with gr.Row(equal_height=False):
                        # ---------- 左：参数 ----------
                        with gr.Column(scale=10, min_width=380):
                            gr.Markdown("#### 参数 · 第 1~4 步")
                            with gr.Group():
                                gr.Markdown("**1 · 模型与产出**")
                                with gr.Row(equal_height=True):
                                    model_in = gr.Textbox(
                                        label="基座模型：HF ID 或本地路径",
                                        value=DEFAULTS.get("model", ""),
                                        placeholder="Qwen/Qwen3-30B-A3B-Instruct-2507、/data/models/Qwen3-30B 或 ~/ckpt/Llama-3.1-8B",
                                        info="HF ID 会下载；本地路径要写在跑任务那侧的机器上，开跑前先确认里面有 config.json",
                                        scale=3)
                                    bs_in = gr.Number(label="batch_size", value=0, precision=0)
                                with gr.Row(equal_height=True):
                                    export_in = gr.Textbox(
                                        label="导出路径（GPU 机器上的绝对路径或 ~）",
                                        placeholder="~/models/qwen-nvfp4", scale=3)
                                    trc_cb = gr.Checkbox(label="trust_remote_code", value=True)
                                dev_dd = gr.Dropdown(["cuda", "cuda:0", "cpu"], value="cuda",
                                                     label="device")

                            with gr.Group():
                                gr.Markdown("**2 · 量化方式**")
                                with gr.Row(equal_height=True):
                                    qf_dd = gr.Dropdown(qf_choices,
                                                        value=DEFAULTS.get("qformat", "nvfp4"),
                                                        label="权重 qformat", container=True,
                                                        scale=3)
                                    kv_dd = gr.Dropdown(kv_choices0,
                                                        value=DEFAULTS.get("kv_cache_qformat",
                                                                           "none"),
                                                        label="KV cache", container=True)
                                with gr.Row(equal_height=True):
                                    recipe_in = gr.Textbox(label="recipe preset", value="",
                                                           placeholder="default / general",
                                                           scale=3)
                                    combo_dd = gr.Dropdown(list(vocab.COMBOS.keys()), value=None,
                                                           label="快速预设", container=True)

                            with gr.Group():
                                gr.Markdown("**3 · 校准数据**")
                                ds_dd = gr.Dropdown(ds_all, value=[],
                                                    label="数据集（内置，可多选）",
                                                    multiselect=True)
                                with gr.Row(equal_height=True):
                                    sizes_dd = gr.Dropdown(ds_all, value=[],
                                                           label="勾选这些设定每集条数",
                                                           multiselect=True, scale=3)
                                    calib_seq_in = gr.Number(
                                        label="calib_seq（校准序列长）",
                                        value=DEFAULTS.get("calib_seq", 512), precision=0)
                                sizes_table = gr.DataFrame(label="条数（改数字）",
                                                           headers=["calib_size"],
                                                           datatype=["number"])
                                custom_df = gr.DataFrame(
                                    label="额外数据集（HF 名 或 GPU 机器上的本地路径）",
                                    headers=["数据集名（HF 名或本机路径）", "calib_size"],
                                    datatype=["str", "number"], interactive=True)
                                calib_img_cb = gr.Checkbox(
                                    label="多模态：校准带图像（VLM + 图文混合集时勾）", value=False)

                            with gr.Accordion("4 · 高级参数（默认即可，点开才生效）",
                                              open=False):
                                with gr.Row(equal_height=True):
                                    attn_in = gr.Textbox(label="attn_implementation", value="",
                                                         placeholder="eager / sdpa / flash_attention_2")
                                    gmp_in = gr.Number(label="gpu_max_mem_percentage",
                                                       value=None, precision=2)
                                with gr.Row(equal_height=True):
                                    fsdp_cb = gr.Checkbox(label="use_fsdp2", value=False)
                                    offload_cb = gr.Checkbox(label="cpu_offload", value=False)
                                    lmm_cb = gr.Checkbox(label="low_memory_mode", value=False)
                                    skip_cb = gr.Checkbox(label="skip_generate", value=False)
                            preview_md = gr.Code(label="将执行的命令", language="shell")

                        # ---------- 右：运行 + 进度 + 日志 ----------
                        with gr.Column(scale=10, min_width=380):
                            gr.Markdown("#### 运行 · 第 5 步")
                            prog_html = gr.HTML(_prog_idle(), elem_classes=["mo-prog"])
                            with gr.Row(equal_height=True):
                                start_btn = gr.Button("▶ 启动量化", variant="primary")
                                abort_btn = gr.Button("■ 中止", variant="stop")
                            proxy_note = gr.Markdown("")
                            log_box = gr.Textbox(label="日志（tail）", lines=18, max_lines=200,
                                                 interactive=False, autoscroll=True)
                            gr.Markdown("日志每 5 秒刷一次。**下载权重**这一步看上面的进度条："
                                        "跑到「加载模型到显存」就说明权重已经下完。")

                # ============================================ 评测 =====
                with gr.Column(visible=False, elem_classes=["mo-pane"]) as pane_eval:  # noqa: SIM117
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=10, min_width=380):
                            gr.Markdown("#### 参数")
                            with gr.Group():
                                gr.Markdown("**1 · 模型（自动填 = 省一次拼写错误）**")
                                ev_q_in = gr.Textbox(label="量化模型（GPU 机器上的路径）",
                                                     placeholder="~/models/qwen3-30b-nvfp4")
                                with gr.Row(equal_height=True):
                                    ev_b_in = gr.Textbox(
                                        label="基线模型（留空 = 只跑量化模型）",
                                        placeholder="Qwen/Qwen3-30B-A3B-Instruct-2507", scale=4)
                                    fetch_b_btn = gr.Button("从 config.json\n自动识别",
                                                            size="sm", scale=1)
                                fetch_b_note = gr.Markdown("")
                            with gr.Group():
                                gr.Markdown("**2 · 部署 + 数据集**")
                                with gr.Row(equal_height=True):
                                    ev_tp = gr.Slider(1, 8, value=1, step=1, label="tensor_parallel")
                                    ev_port = gr.Number(label="port", value=8000, precision=0)
                                    ev_batch = gr.Textbox(label="batch_size", value="auto",
                                                          scale=2)
                                with gr.Row(equal_height=True):
                                    ev_preset = gr.Dropdown(list(EVAL_PRESETS.keys()),
                                                            value="快速检查（GPQA 单项）",
                                                            label="预设", scale=3)
                                    ev_fetch = gr.Button("拉取历史结果", size="sm")
                                ev_tasks = gr.Dropdown(sorted(EVAL_TASKS), value=[],
                                                       label="数据集（可多选）", multiselect=True)
                        with gr.Column(scale=10, min_width=380):
                            gr.Markdown("#### 运行")
                            ev_prog = gr.HTML(_prog_idle("等待评测"), elem_classes=["mo-prog"])
                            with gr.Row(equal_height=True):
                                ev_run = gr.Button("▶ 启动评测", variant="primary")
                                ev_abort = gr.Button("■ 中止", variant="stop")
                            ev_note = gr.Markdown("")
                            ev_log = gr.Textbox(label="日志（tail）", lines=14, max_lines=200,
                                                interactive=False, autoscroll=True)
                            ev_result_md = gr.Markdown("")
                            ev_table = gr.DataFrame(headers=["模型", "任务", "指标", "量化",
                                                             "基线", "Δ(量化-基线)"],
                                                    datatype=["str", "str", "str", "str",
                                                              "str", "str"])

                # ======================================== 结果对比 =====
                with gr.Column(visible=False, elem_classes=["mo-pane"]) as pane_recipe:  # noqa: SIM117
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=10, min_width=380):
                            gr.Markdown("#### 1 · 取本次导出的量化配方\n"
                                        "读最近一次任务日志里的 `--export_path` 与 `layer_config.json`，"
                                        "对比 ignore / 量化层分布。")
                            with gr.Row(equal_height=True):
                                validate_btn = gr.Button("读取本次导出配方", variant="primary",
                                                         scale=2)
                                official_in = gr.Textbox(
                                    label="官方 HF 仓（留空 = 不对比）", scale=3,
                                    value="nvidia/Qwen3.8-27B-NVFP4")
                            result_md = gr.Markdown("")
                        with gr.Column(scale=10, min_width=380):
                            gr.Markdown("#### 2 · 分组明细与差异")
                            group_table = gr.DataFrame(headers=["group", "bits", "targets 数",
                                                                "代表 targets"],
                                                       datatype=["str", "str", "number", "str"])
                            diff_md = gr.Markdown("")
                            files_md = gr.Markdown("")

                # ======================================== 连接设置 =====
                with gr.Column(visible=False, elem_classes=["mo-pane"]) as pane_conn:
                    gr.Markdown("WebUI 就跑在这台机器上，下面的路径都按**本机**解释。"
                                "改完点保存，立刻重新探测。")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=10, min_width=340):
                            gr.Markdown("#### 运行环境")
                            env_note = gr.Markdown("")
                            ed_repo = gr.Textbox(label="Model-Optimizer 仓库",
                                                 value=CFG.local.repo,
                                                 placeholder=DEFAULT_REPO)
                            ed_venv = gr.Textbox(
                                label="venv 目录（相对项目根 = 项目内）",
                                value=CFG.local.venv, placeholder=DEFAULT_VENV,
                                info="首次用点量化区的「创建 venv」，装完会写在这里")
                            ed_task = gr.Textbox(label="任务与产物目录",
                                                 value=CFG.local.task_dir,
                                                 placeholder=DEFAULT_TASK_DIR,
                                                 info="产物是 GB 级，建议留在仓库外")
                            ed_cache = gr.Textbox(label="HuggingFace 缓存",
                                                  value=CFG.local.hf_cache,
                                                  placeholder=DEFAULT_HF_CACHE)
                            with gr.Row(equal_height=True):
                                save_env_btn = gr.Button("💾 保存路径",
                                                         variant="primary", size="sm")
                                venv_env_btn = gr.Button("创建 venv", size="sm")
                        with gr.Column(scale=10, min_width=340):
                            gr.Markdown("#### HuggingFace 下载走哪条网")
                            gr.Markdown("代理地址**一律手填**，程序不猜拓扑。"
                                        "填的是这台机器自己能连到的地址，"
                                        "如 `http://127.0.0.1:7890`。")
                            net_mode = gr.Radio(PROXY_MODES, value=CFG.net.mode, label="模式")
                            net_proxy = gr.Textbox(label="代理地址",
                                                   value=CFG.net.proxy,
                                                   placeholder="http://127.0.0.1:7890")
                            with gr.Row(equal_height=True):
                                save_net_btn = gr.Button("💾 保存代理", size="sm",
                                                         variant="primary")
                                test_net_btn = gr.Button("测试连通性", size="sm")
                            net_note = gr.Markdown("")


        nav.change(
            lambda v: [gr.update(visible=v == "量化"),
                       gr.update(visible=v == "评测"),
                       gr.update(visible=v == "对配方"),
                       gr.update(visible=v == "连接设置")],
            nav, [pane_quant, pane_eval, pane_recipe, pane_conn])
        # ============================================================ 状态
        qf_val_st = gr.State(DEFAULTS.get("qformat", "nvfp4"))
        kv_val_st = gr.State(DEFAULTS.get("kv_cache_qformat", "none"))
        task_state = gr.State(None)
        eval_state = gr.State(None)

        all_inputs = [model_in, trc_cb, qf_dd, recipe_in, kv_dd, ds_dd,
                      sizes_dd, sizes_table, custom_df, calib_seq_in, calib_img_cb,
                      export_in, bs_in, dev_dd, attn_in, fsdp_cb, offload_cb, lmm_cb,
                      gmp_in, skip_cb]

        # ------------------------------------------------------ 通用工具
        def do_save_env(repo, venv, task_dir, hf_cache):
            """保存运行环境路径并热更新 CFG；返回 (提示, 顶栏环境行)。"""
            try:
                CFG.local.repo = (repo or "").strip() or "~/Model-Optimizer"
                CFG.local.venv = (venv or "").strip() or ".venvs/modelopt"
                CFG.local.task_dir = (task_dir or "").strip() or "~/quant-webui-tasks"
                CFG.local.hf_cache = (hf_cache or "").strip() or "~/.cache/huggingface"
                save_local(CFG.local)
            except Exception as ex:
                return gr.update(value=f"⚠ 保存失败：{ex}"), _env_line()
            return gr.update(value="✅ 已保存（立即生效，无需重启）"), _env_line()

        def do_save_net(mode, proxy):
            """保存 HF 网络设置并热更新 CFG；返回 (提示, 顶栏网络行)。"""
            try:
                CFG.net = NetConfig(mode=(mode or PROXY_AUTO).strip(),
                                    proxy=(proxy or "").strip())
                save_net(CFG.net)
            except Exception as ex:
                return gr.update(value=f"⚠ 保存失败：{ex}"), _net_line()
            return gr.update(value="✅ 已保存（立即生效）"), _net_line()

        def do_test_net():
            """按当前已保存的网络设置实测一次 HF 连通性。"""
            try:
                ok, detail = runner.test_proxy(CFG)
                return ("✅ 200/2xx 通过" if ok else "⚠ 未通过") + f"：{detail}"
            except Exception as ex:
                return f"⚠ 测试失败：{type(ex).__name__}: {ex}"

        def do_refresh(qf_cur, kv_cur):
            """刷新状态栏 + 重新扫描本机 qformat / kv 词表。"""
            st = runner.probe_host(CFG)
            md = "**本机状态**\n"
            if st.error:
                md += f"⚠ {st.error}\n\n"
            md += " | ".join(f"{k}: {v}" for k, v in st.as_dict().items())
            try:
                qf = vocab.scan_qformats(CFG.local.repo)
            except Exception as ex:
                qf = {"model": vocab.all_qformats(), "kv": vocab.KV_QFORMATS,
                      "source": "内置（扫描词表失败）", "error": str(ex)}
            model_raw = qf.get("model")
            model_list = [str(v) for v in model_raw] if isinstance(model_raw, list) \
                else vocab.all_qformats()
            qf_choices = vocab.render_qformat_dropdown(model_list)
            if qf_cur not in qf_choices:
                qf_cur = DEFAULTS.get("qformat", "nvfp4")
            kv_choices = ["── 内置 ──"] + vocab.kv_choices()
            kv_raw = qf.get("kv")
            for v in ([str(x) for x in kv_raw] if isinstance(kv_raw, list) else []):
                if v not in vocab.KV_QFORMATS and v not in kv_choices:
                    kv_choices += ["── 机器预设 ──", v] if kv_choices[-1] != "── 机器预设 ──" else [v]
            if kv_cur not in kv_choices:
                kv_cur = DEFAULTS.get("kv_cache_qformat", "none")
            md += f"\n\n词表来源: {qf['source']}"
            return (md, gr.update(choices=qf_choices, value=qf_cur), qf_cur,
                    gr.update(choices=kv_choices, value=kv_cur), kv_cur)

        def initial_probe():
            """启动时填充顶栏两行 + 状态栏占位 + 词表默认值。"""
            return (_env_line(), _net_line(), "**正在检测本机…**",
                    DEFAULTS.get("qformat", "nvfp4"), gr.update(),
                    DEFAULTS.get("kv_cache_qformat", "none"), gr.update())

        def refresh_preview(*a):
            try:
                cmd = build_cmd(build_params(list(a)))
                return _prettify_cmd(cmd)
            except Exception as ex:
                return f"（拼参出错: {ex}）"

        def sizes_rows(names):
            return [[1024] for _ in (names or [])] if names else None

        # -------------------------------------------------- 量化任务
        def do_start(*a):
            try:
                p = build_params(list(a))
            except Exception as ex:
                yield (f"⚠ {ex}", _prog_html(progress.ProgressReport(
                    error=str(ex))), "参数有误", None)
                return
            cmd = build_cmd(p)
            st = runner.probe_host(CFG)
            if not st.reachable:
                yield (f"⚠ 目标不可达: {st.error}", _prog_html(progress.ProgressReport(
                    error=st.error or "不可达")), "启动前检查失败", None)
                return
            if not st.venv_ok:
                yield ("⚠ venv 未创建", _prog_html(progress.ProgressReport(
                    error="venv 未创建")), "请先点上方「一键创建 venv」", None)
                return
            model = str(p.get("model") or "").strip()
            if is_local_model(model):
                why = check_local_model(model)
                if why:
                    yield ("⚠ 模型路径有问题", _prog_html(progress.ProgressReport(error=why)),
                           why, None)
                    return
            gpu_warn = ""
            if st.gpu_busy:
                gpu_warn = (f"⚠ GPU 当前 {st.gpu_util}% 占用中：不抢占现有任务，"
                            f"会和它抢显存。确认那是无用进程再启动（「■ 中止」能停自己的）。\n")
            note = gpu_warn + f"任务目录：{CFG.local.task_dir}\n"
            yield note, _prog_html(progress.ProgressReport(stage_idx=0,
                                                           stage="提交任务")), "任务提交中…", None
            try:
                task = runner.start_task(CFG, CFG.local.task_dir,
                                         task_slug(model),
                                         str(p.get("export_path") or ""),
                                         cmd, runner.proxy_for(CFG), lambda s: None)
            except Exception as ex:
                yield (note, _prog_html(progress.ProgressReport(error=str(ex))),
                       f"启动失败: {ex}", None)
                return
            while True:
                runner.poll_task(task)
                raw = runner.read_log(task, 30000)
                rep = progress.parse(raw)
                yield (note, _prog_html(rep), raw, task)
                if task.state != "running":
                    mark = {"done": "✅ 任务结束", "failed": "❌ 任务失败",
                            "aborted": "⛔ 已中止"}.get(task.state, task.state)
                    final = progress.parse(runner.read_log(task, 60000))
                    yield (mark + "\n" + note, _prog_html(final),
                           runner.read_log(task), task)
                    return
                time.sleep(5)

        def do_abort(state):
            if state is None:
                return "没有运行中的任务"
            return runner.abort_task(state)

        # -------------------------------------------------- 环境准备
        def do_venv():
            buf: list[str] = []
            try:
                runner.ensure_venv(lambda s: buf.append(s), CFG)
            except Exception as ex:
                return f"⚠ 创建失败：{ex}"
            text = "\n".join(buf) or "（无输出）"
            rep = progress.parse_install(buf)
            head = "✅ venv 就绪" if not rep.error else f"⚠ {rep.error}"
            return f"{head} · {rep.detail}\n```\n{text[-4000:]}\n```"

        def do_eval_env():
            buf: list[str] = []
            try:
                runner.ensure_eval_env(lambda s: buf.append(s), CFG)
            except Exception as ex:
                return f"⚠ 安装失败：{ex}"
            text = "\n".join(buf) or "（无输出）"
            rep = progress.parse_install(buf)
            head = "✅ 评测环境就绪" if not rep.error else f"⚠ {rep.error}"
            return f"{head} · {rep.detail}\n```\n{text[-4000:]}\n```"

        # -------------------------------------------------- 结果校验
        def do_validate(state, official):
            if state is None:
                return "没有已提交的任务（先跑一个任务）", None, "（无数据）", ""
            log = runner.read_log(state, 5000)
            hint = ""
            with contextlib.suppress(Exception):
                hint = state.remote_cmd.split("--export_path")[-1].split()[0].strip("'\"")
            mine = validate.scan_export_dir(log, export_dir_hint=hint)
            if mine.error:
                return f"⚠ {mine.error}", None, "（无数据）", ""
            md = (f"**{mine.source}**\nquant_method: {mine.quant_method}\n"
                  f"ignore: {validate.ignore_preview(mine)}")
            diff = ""
            if official.strip():
                try:
                    off = validate.fetch_official_summary(
                        official, proxy=runner.proxy_for(CFG))
                    diff = "\n\n".join(validate.diff_summaries(mine, off))
                except Exception as ex:
                    diff = f"⚠ 官方对照失败: {ex}"
            files = ""
            if mine.files:
                files = f"\n\n**导出目录文件**\n```\n{mine.files}\n```"
            return md, validate.to_table_rows(mine), diff, files

        # -------------------------------------------------- 评测
        def do_fetch_baseline(quant_path):
            q = (quant_path or "").strip()
            if not q:
                return "（先填量化模型路径）", gr.update()
            r, out, _ = runner.run_shell(f"cat {q_remote(q)}/config.json 2>/dev/null",
                                         timeout=20)
            if r != 0 or not out.strip():
                return "⚠ 读不到量化 checkpoint 的 config.json（检查上面路径）", gr.update()
            try:
                b = (json.loads(out).get("quantization_config", {})
                     .get("_name_or_path", "") or
                     json.loads(out).get("_name_or_path", ""))
            except Exception:
                return "⚠ config.json 解析失败", gr.update()
            if not b or b == q:
                return "⚠ config.json 里没有基线模型名，请手填", gr.update()
            return f"✅ 已识别基线：`{b}`", gr.update(value=b)

        def do_ev_preset(name):
            if not name:
                return []
            return list(EVAL_PRESETS.get(name, []))

        def do_eval(quant, baseline, tp, port, batch, tasks):
            ok, msg = runner.probe_eval_env(CFG)
            if not ok:
                yield (f"⚠ {msg}", _prog_html(progress.ProgressReport(error=msg)),
                       "请先点上方「一键装评测环境」（vllm + lm_eval）", None)
                return
            note = f"任务目录：{CFG.local.task_dir}/eval_*\n"
            yield note, _prog_idle("启动评测"), "评测启动中…", None
            try:
                script = build_eval_script(quant, baseline, tasks,
                                           int(tp or 1), int(port or 8000),
                                           str(batch or "auto"))
                ts = time.strftime("%Y%m%d-%H%M%S")
                tdir = str(CFG.local.task_path / f"eval_{ts}")
                task = runner.start_script_task(CFG, tdir, script,
                                                runner.proxy_for(CFG),
                                                lambda s: None, pkill_pattern="vllm serve")
            except Exception as ex:
                yield (note, _prog_html(progress.ProgressReport(error=str(ex))),
                       f"启动失败: {ex}", None)
                return
            while True:
                runner.poll_task(task)
                raw = runner.read_log(task, 30000)
                yield note, _prog_html(_ev_progress(raw)), raw, task
                if task.state != "running":
                    mark = {"done": "✅ 评测结束", "failed": "❌ 评测失败",
                            "aborted": "⛔ 已中止"}.get(task.state, task.state)
                    yield (mark + "\n" + note, _prog_html(_ev_progress(
                        runner.read_log(task, 60000))), runner.read_log(task), task)
                    return
                time.sleep(5)

        def do_ev_abort(state):
            if state is None:
                return "没有运行中的评测"
            return runner.abort_task(state) + "\n（已连 vllm serve 一起停）"

        def do_fetch_results():
            results = runner.fetch_eval_results(str(CFG.local.task_path))
            if not results:
                return "（没找到 results_*.json）", None
            rows, note = _result_tables(results)
            return f"**{len(results)} 个结果文件**\n{note}", rows

        # ============================================================ 绑定
        for inp in all_inputs[1:]:
            inp.change(refresh_preview, all_inputs, preview_md)
        combo_dd.change(lambda c: list(vocab.COMBOS.get(c, [])) if c else [], combo_dd, ds_dd)
        sizes_dd.change(sizes_rows, sizes_dd, sizes_table)

        # ------------------------------------------------ 状态 / 量化 / 评测
        refresh_btn.click(do_refresh, [qf_val_st, kv_val_st],
                          [status_md, qf_dd, qf_val_st, kv_dd, kv_val_st])
        ds_dd.change(lambda d: d or [], ds_dd, sizes_dd)
        start_btn.click(do_start, all_inputs,
                        [proxy_note, prog_html, log_box, task_state])
        abort_btn.click(do_abort, task_state, proxy_note)
        validate_btn.click(do_validate, [task_state, official_in],
                           [result_md, group_table, diff_md, files_md])
        venv_btn.click(do_venv, None, venv_log)
        eval_env_btn.click(do_eval_env, None, venv_log)

        fetch_b_btn.click(do_fetch_baseline, ev_q_in, [fetch_b_note, ev_b_in])
        ev_preset.change(do_ev_preset, ev_preset, ev_tasks)
        ev_run.click(do_eval, [ev_q_in, ev_b_in, ev_tp, ev_port, ev_batch, ev_tasks],
                     [ev_note, ev_prog, ev_log, eval_state])
        ev_abort.click(do_ev_abort, eval_state, ev_note)
        ev_fetch.click(do_fetch_results, None, [ev_result_md, ev_table])

        # ------------------------------------------------ 运行环境 / HF 网络
        save_env_btn.click(do_save_env, [ed_repo, ed_venv, ed_task, ed_cache],
                           [env_note, mode_md])
        venv_env_btn.click(do_venv, None, venv_log)
        save_net_btn.click(do_save_net, [net_mode, net_proxy], [net_note, net_md])
        test_net_btn.click(do_test_net, None, net_note)

        app.load(initial_probe, None,
                 [mode_md, net_md, status_md, qf_val_st, qf_dd, kv_dd, kv_val_st])


    return app

def _prettify_cmd(cmd: str) -> str:
    """单行命令 → 反斜杠多行（仅展示）。"""
    head, _, tail = cmd.partition(" && ")
    if not tail:
        return cmd
    args = tail.split(" ")
    if len(args) <= 2:
        return cmd
    first, rest = args[0], args[1:]
    lines = [head + " && " + first + " \\"]
    for i, arg in enumerate(rest):
        lines.append("    " + arg + (" \\" if i < len(rest) - 1 else ""))
    return "\n".join(lines)


def main() -> None:
    _build_app().launch(server_name=CFG.server_host, server_port=CFG.server_port,
                        theme=_theme(), css=MO_CSS)


def _theme() -> gr.Theme:
    return gr_themes.Ocean(
        primary_hue="emerald", secondary_hue="sky",
        font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
    )


if __name__ == "__main__":
    main()

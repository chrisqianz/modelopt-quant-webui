"""ModelOpt 量化 WebUI 主入口（Gradio）。

布局：顶部 host + 状态栏 → [量化] tab（模型 → 量化配置 → 校准数据集 →
高级参数 → 命令预览 → 启动/日志 → 结果校验）→ [评测] tab（vLLM 部署 +
lm-eval 跑「量化 vs 基线」对比，回答"量化有没有吃瘪"）。

机器无关：既能从开发机 SSH 驱动 GPU 机器，也能整目录搬到 GPU 机器上直接跑（
机器模式 = 本机）。路径与代理地址一律在 UI 里填，代码里不写死。
"""

from __future__ import annotations

import json
import re
import shlex
import time

import gradio as gr

from webui import runner, validate, vocab
from webui import progress
from webui.config import (
    DEFAULT_HF_CACHE, DEFAULT_REPO, DEFAULT_TASK_DIR, DEFAULT_VENV, EXEC_MODES,
    MODE_LOCAL, MODE_REMOTE, PROXY_AUTO, PROXY_DIRECT, PROXY_MANUAL, PROXY_MODES,
    HostConfig, NetConfig, hosts_from_dicts, load_config, save_hosts, save_net)
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
    """把事件回调位置参数解析成 build_remote_cmd 需要的 dict。
    a[0] = host_name，a[1..20] 对应 UI 参数区固定顺序（见 INPUT_ORDER）。"""
    sizes: dict[str, int] = {}
    for i, name in enumerate(a[7] or []):  # sizes_names 勾选顺序
        row = a[8][i] if a[8] and i < len(a[8]) else None
        if row and row[0] is not None:
            sizes[name] = int(row[0])
    # gradio 6 前端有时会把 (展示名, 值) 对整体发回（多一层 list），取最后一项兜底
    qf = a[3]
    if isinstance(qf, (list, tuple)) and qf:
        qf = qf[-1]
    kv = a[5]
    if isinstance(kv, (list, tuple)) and kv:
        kv = kv[-1]
    return {
        "model": a[1], "trust_remote_code": a[2], "qformat": qf,
        "recipe": a[4], "kv_cache_qformat": kv, "datasets": a[6],
        "dataset_sizes": sizes, "custom_datasets": a[9],
        "calib_seq": a[10], "calib_with_images": a[11], "export_path": a[12],
        "batch_size": a[13], "device": a[14], "attn_implementation": a[15],
        "use_fsdp2": a[16], "cpu_offload": a[17], "low_memory_mode": a[18],
        "gpu_max_mem_percentage": a[19], "skip_generate": a[20],
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


def check_local_model(host, model: str) -> str:
    """本地模型跑前在目标机上确认一下，别等几分钟后才告诉你路径写错了。
    返回 "" = 没问题，否则是给用户看的错误说明。"""
    probe = (f"P={runner.q_remote(model, host)}; "
             'if [ -d "$P" ]; then '
             'if [ -f "$P/config.json" ]; then echo __OK__; else echo __NOCFG__; fi; '
             "else echo __NODIR__; fi")
    _, flag, _ = runner.run_remote(host, probe, timeout=30)
    flag = flag or ""
    if "__OK__" in flag:
        return ""
    if "__NODIR__" in flag:
        return (f"{host.where}上没有这个目录：{model}\n"
                "路径要写在执行那一侧的机器上（本机模式=这台，远程模式=远程机）。")
    return f"{model} 在，但里面没有 config.json，不像 HF 模型目录。"


def build_remote_cmd(host: runner.HostConfig, p: dict) -> str:
    """把 UI 参数拼成远程 hf_ptq.py 命令（含 cd 与 venv python）。
    会被包进 nohup bash -c '...'，所以 ~ 前缀路径一律走 q_remote 转绝对路径。"""
    repo = q_remote(host.remote_repo, host)
    venv = q_remote(host.venv_dir, host)
    parts = [f"cd {repo} && {venv}/bin/python examples/hf_ptq/hf_ptq.py"]

    model = (p.get("model") or "").strip()
    if not model:
        raise gr.Error("请填写模型路径/名称")
    parts.append(f"--pyt_ckpt_path "
                 f"{q_remote(model, host) if is_local_model(model) else shlex.quote(model)}")
    if p.get("trust_remote_code"):
        parts.append("--trust_remote_code")

    qfmt = (p.get("qformat") or "").strip()
    recipe = (p.get("recipe") or "").strip()
    if recipe:
        parts.append(f"--recipe {q_remote(recipe, host) if recipe.startswith(('/', '~')) else shlex.quote(recipe)}")
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
        parts.append(f"--export_path {q_remote(export, host) if export.startswith(('/', '~')) else shlex.quote(export)}")

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

def _ref_path(host: runner.HostConfig, s: str) -> str:
    """远程路径引用：~ 转绝对，其余 shlex.quote。"""
    return q_remote(s, host) if s.startswith(("/", "~")) else shlex.quote(s)


def build_eval_script(host: runner.HostConfig, quant: str, baseline: str,
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

def _mode_line(name: str) -> str:
    """顶栏那颗「跑在哪」的胶囊。"""
    try:
        h = CFG.get_host(name)
    except ValueError:
        return "**执行位置**：未配置"
    if h.is_local:
        return f"**执行位置**：本机（UI 就跑在这台机器上）"
    return f"**执行位置**：SSH → `{h.ssh}:{h.ssh_port}`（{h.auth_kind}）"


def _net_line(host_name: str = "") -> str:
    """顶栏那颗「HF 下载走哪」的胶囊（不探测，只说当前配置）。"""
    net = CFG.net
    if net.mode == PROXY_DIRECT:
        return "**HF 下载**：直连（不注入代理）"
    url, note = "", ""
    try:
        url, note = runner.proxy_url_for(CFG, CFG.get_host(host_name))
    except Exception:                # 没配机器时用配置值本身说事
        pass
    use = url or net.remote_proxy or net.local_proxy
    if net.mode == PROXY_MANUAL:
        return f"**HF 下载**：强制代理 `{use or '⚠ 还没填地址'}`" + (f"（{note}）" if note else "")
    if use:
        return f"**HF 下载**：自动 · 直连优先，不通走 `{use}`"
    return "**HF 下载**：自动 · 直连优先（未填代理地址，直连不通就下不动）"


def _host_rows() -> list[dict]:
    return [h.to_dict() for h in CFG.hosts]


def _host_list_md(hosts: list[HostConfig]) -> str:
    if not hosts:
        return "（还没有机器。下面填一台。）"
    lines = ["| 机器 | 执行位置 | 认证 | 仓库 |", "|---|---|---|---|"]
    for h in hosts:
        repo = h.remote_repo or ("—" if h.is_local else "（未填）")
        lines.append(f"| **{h.name}** | {h.where} | {h.auth_kind} | `{repo}` |")
    return "\n".join(lines)


def _host_form(h: HostConfig | None):
    """把表单填成某台机器的现值（h=None = 新增空白）。密码框一律留空。"""
    v = {} if h is None else dict(name=h.name, mode=h.mode, ssh=h.ssh,
                                  ssh_port=h.ssh_port, remote_repo=h.remote_repo,
                                  venv_dir=h.venv_dir, task_dir=h.task_dir,
                                  hf_cache=h.hf_cache)
    get = lambda k, dflt: gr.update(value=v.get(k, dflt))  # noqa: E731
    return (get("name", ""), get("mode", MODE_LOCAL), get("ssh", ""),
            get("ssh_port", 22), gr.update(value=""), get("remote_repo", DEFAULT_REPO),
            get("venv_dir", DEFAULT_VENV), get("task_dir", DEFAULT_TASK_DIR),
            get("hf_cache", DEFAULT_HF_CACHE),
            gr.update(interactive=True))


def _apply_hosts(hosts: list[HostConfig], selected: str) -> str:
    """保存后把机器灌回 CFG，让所有用 CFG.get_host 的回调立刻看到新值。"""
    CFG.hosts = hosts
    names = [h.name for h in hosts]
    if selected not in names:
        selected = names[0] if names else ""
    return selected




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
    names0 = CFG.host_names()
    first0 = names0[0] if names0 else ""
    try:
        host0 = CFG.get_host(first0)
    except ValueError:
        host0 = None

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
                host_dd = gr.Dropdown(names0, value=first0, label="目标机器")
                refresh_btn = gr.Button("⟳ 探测", variant="secondary", size="sm")
                mode_md = gr.Markdown(_mode_line(first0))
                net_md = gr.Markdown(_net_line(first0))
                venv_btn = gr.Button("一键创建 venv", size="sm")
                eval_env_btn = gr.Button("一键装评测环境", size="sm")
                eval_env_st = gr.State("（未探测）")
            with gr.Column(scale=1, min_width=0):
                status_md = gr.Markdown("点 **⟳ 探测** 看 GPU / venv / 磁盘 / HF 网络实测情况。")
                venv_log = gr.Markdown("", elem_classes=["mo-log"])

                # ============================================ 量化 =====
                with gr.Column(visible=True, elem_classes=["mo-pane"]) as pane_quant:
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
                                    headers=["数据集名（HF 名或远程本地路径）", "calib_size"],
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
                with gr.Column(visible=False, elem_classes=["mo-pane"]) as pane_eval:
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
                with gr.Column(visible=False, elem_classes=["mo-pane"]) as pane_recipe:
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
                    gr.Markdown("机器与代理配置存在 `hosts.json`（保存后 **chmod 600**，"
                                "只有当前用户可读）。**密码只用于连接这台机器，不进日志、"
                                "不进命令、不进 git。**")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=10, min_width=340):
                            gr.Markdown("#### 机器清单")
                            host_list_md = gr.Markdown(_host_list_md(CFG.hosts))
                            with gr.Row(equal_height=True):
                                pick_dd = gr.Dropdown(names0, value=first0,
                                                      label="选择要改的机器", container=True,
                                                      scale=3)
                                new_btn = gr.Button("➕ 新增", size="sm")
                            with gr.Row(equal_height=True):
                                test_btn = gr.Button("测试连接", size="sm", scale=2)
                                del_btn = gr.Button("删除这台", variant="stop", size="sm",
                                                    scale=2)
                            host_note = gr.Markdown("")
                        with gr.Column(scale=12, min_width=380):
                            gr.Markdown("#### 编辑机器")
                            ed_name = gr.Textbox(label="名称（下拉里显示这个）",
                                                 placeholder="5090 工作站 / 本机")
                            ed_mode = gr.Radio(EXEC_MODES, value=MODE_LOCAL, label="执行位置")
                            with gr.Row(equal_height=True):
                                ed_ssh = gr.Textbox(label="SSH 目标 user@host",
                                                    placeholder="user@192.168.1.20", scale=3)
                                ed_port = gr.Number(label="端口", value=22, precision=0)
                            ed_pass = gr.Textbox(label="SSH 密码（留空 = 用密钥/ssh-agent）",
                                                 type="password",
                                                 info="sshpass 自动登录：密码不写进命令行，"
                                                      "也不会出现在日志里")
                            with gr.Accordion("路径（留空用默认；本机模式按 UI 所在机器解释）",
                                              open=False):
                                ed_repo = gr.Textbox(label="ModelOpt 仓库",
                                                     placeholder=DEFAULT_REPO)
                                ed_venv = gr.Textbox(label="venv 目录",
                                                     placeholder=DEFAULT_VENV)
                                ed_task = gr.Textbox(label="任务目录",
                                                     placeholder=DEFAULT_TASK_DIR)
                                ed_cache = gr.Textbox(label="HF 缓存",
                                                      placeholder=DEFAULT_HF_CACHE)
                            save_host_btn = gr.Button("💾 保存机器清单", variant="primary")

                    gr.Markdown("---")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=10, min_width=340):
                            gr.Markdown("#### HuggingFace 下载走哪条网")
                            gr.Markdown("代理地址**一律手填**，程序不猜也不写死。"
                                        "填的是 **HF_HTP_PROXY 要用的地址**，"
                                        "必须从「执行位置」那台机器可达：\n"
                                        "- 本机模式 → 填这台机器自己的，如 `http://127.0.0.1:7890`\n"
                                        "- SSH 远程 → 填远程能访问的地址（比如开发机上监听的代理，"
                                        "要写局域网 IP，`127.0.0.1` 在远程指的是远程自己）")
                            net_mode = gr.Radio(PROXY_MODES, value=CFG.net.mode, label="模式")
                            net_local = gr.Textbox(label="本机的代理地址（UI 所在机器）",
                                                   value=CFG.net.local_proxy,
                                                   placeholder="http://127.0.0.1:7890")
                            net_remote = gr.Textbox(label="远程机器的代理地址（远程可达）",
                                                    value=CFG.net.remote_proxy,
                                                    placeholder="http://192.168.1.10:7890")
                            with gr.Row(equal_height=True):
                                save_net_btn = gr.Button("💾 保存代理配置", size="sm",
                                                         variant="primary")
                                test_net_btn = gr.Button("测试代理连通性", size="sm")
                            net_note = gr.Markdown("")
                        with gr.Column(scale=10, min_width=340):
                            gr.Markdown("#### SSH 连接复用")
                            gr.Markdown("首次连接会用 ControlMaster 建一条主连接（密码只在这一刻用），"
                                        "之后的探测 / 提交 / tail 全复用，**不再重复认证**，"
                                        "也不会触发服务器上的爆破检测。换密码或换机器后点下面重建。")
                            recon_btn = gr.Button("断开复用连接（重连）", size="sm")
                            conn_note = gr.Markdown("")


        nav.change(
            lambda v: [gr.update(visible=v == "量化"),
                       gr.update(visible=v == "评测"),
                       gr.update(visible=v == "对配方"),
                       gr.update(visible=v == "连接设置")],
            nav, [pane_quant, pane_eval, pane_recipe, pane_conn])
        # ============================================================ 状态
        qf_val_st = gr.State(DEFAULTS.get("qformat", "nvfp4"))
        kv_val_st = gr.State(DEFAULTS.get("kv_cache_qformat", "none"))
        hosts_st = gr.State(_host_rows())
        task_state = gr.State(None)
        eval_state = gr.State(None)

        all_inputs = [host_dd, model_in, trc_cb, qf_dd, recipe_in, kv_dd, ds_dd,
                      sizes_dd, sizes_table, custom_df, calib_seq_in, calib_img_cb,
                      export_in, bs_in, dev_dd, attn_in, fsdp_cb, offload_cb, lmm_cb,
                      gmp_in, skip_cb]

        # ------------------------------------------------------ 通用工具
        def _hh(items) -> list[HostConfig]:
            return hosts_from_dicts(items or [])

        def refresh_net(host_name):
            return gr.update(value=_net_line(host_name))

        def mode_line(host_name):
            return gr.update(value=_mode_line(host_name))

        # -------------------------------------------------- 机器 CRUD
        def host_list_md_of(items):
            try:
                return _host_list_md(_hh(items))
            except Exception as ex:
                return f"⚠ {ex}"

        def pick_host(name, items):
            """选中一台 → 表单填上它的现值（密码框始终留空）。"""
            try:
                hosts = {h.name: h for h in _hh(items)}
            except Exception:
                hosts = {}
            return _host_form(hosts.get(name))

        def new_host():
            return _host_form(None)

        def toggle_ssh(mode):
            show = gr.update(visible=mode == MODE_REMOTE)
            return show, show, show

        def save_host(editing, name, mode, ssh, port, pwd, repo, venv, task, cache, items):
            """把表单落成清单里的一行（editing 空 = 新增），然后整表落盘。"""
            name = (name or "").strip()
            if not name:
                return (gr.update(), gr.update(), gr.update(),
                        "⚠ 名称不能为空", gr.update())
            rows = [dict(d) for d in (items or [])]
            entry = {"name": name, "mode": mode or MODE_LOCAL, "ssh": (ssh or "").strip(),
                     "ssh_port": port or 22, "password": pwd or "",
                     "remote_repo": (repo or "").strip(), "venv_dir": (venv or "").strip(),
                     "task_dir": (task or "").strip(), "hf_cache": (cache or "").strip()}
            hit = next((i for i, d in enumerate(rows)
                        if str(d.get("name") or "").strip() == (editing or name)), None)
            if hit is None:
                if any(str(d.get("name") or "").strip() == name for d in rows):
                    return (gr.update(), gr.update(), gr.update(),
                            f"⚠ 已有同名机器：{name}", gr.update())
                rows.append(entry)
                action = "已新增"
            else:
                if not (pwd or "").strip() and not (rows[hit].get("password") or ""):
                    pass                      # 两条都空 = 明确用密钥
                elif not (pwd or "").strip():
                    entry["password"] = rows[hit]["password"]   # 没重填 = 保留旧密码
                rows[hit] = entry
                action = "已更新"
            try:
                hosts = _hh(rows)
            except Exception as ex:
                return (gr.update(), gr.update(), gr.update(), f"⚠ {ex}", gr.update())
            save_hosts(hosts)
            sel = _apply_hosts(hosts, editing or name)
            note = (f"✅ {action} **{name}**（{hosts_from_dicts([entry])[0].where} · "
                    f"{hosts_from_dicts([entry])[0].auth_kind}）→ hosts.json（权限 600）")
            return (gr.update(choices=[h.name for h in hosts], value=sel),
                    gr.update(choices=[h.name for h in hosts], value=name),
                    host_list_md_of(hosts), note,
                    gr.update(value=[h.to_dict() for h in hosts]))

        def del_host(name, items):
            rows = [dict(d) for d in (items or [])
                    if str(d.get("name") or "").strip() != (name or "").strip()]
            if len(rows) == len(items or []):
                return (gr.update(), gr.update(), gr.update(),
                        "⚠ 先在上方选中要删除的机器", gr.update())
            try:
                hosts = _hh(rows)
            except Exception as ex:
                return (gr.update(), gr.update(), gr.update(),
                        f"⚠ 至少保留一台机器（{ex}）", gr.update())
            save_hosts(hosts)
            sel = _apply_hosts(hosts, "")
            return (gr.update(choices=[h.name for h in hosts], value=sel),
                    gr.update(choices=[h.name for h in hosts], value=""),
                    host_list_md_of(hosts), f"🗑 已删除 {name}",
                    gr.update(value=[h.to_dict() for h in hosts]))

        def test_conn(name):
            try:
                host = CFG.get_host(name)
            except ValueError as ex:
                return f"⚠ {ex}"
            if host.is_local:
                ok, text = runner.check_connection(host)
                return ("✅ 本机可用：" if ok else "⚠ 本机检查：") + text
            err = runner.ensure_master(host, force=True)
            if err:
                return f"⚠ {err}"
            ok, text = runner.check_connection(host)
            return ("✅ " if ok else "⚠ ") + text

        def reconnect(name):
            try:
                host = CFG.get_host(name)
            except ValueError as ex:
                return f"⚠ {ex}"
            runner.drop_master(host)
            return f"已断开 {host.identity} 的复用连接，下次操作会重连。"

        # -------------------------------------------------- 代理配置
        def net_of(mode, local, remote):
            return NetConfig(mode=mode or PROXY_AUTO,
                             remote_proxy=(remote or "").strip(),
                             local_proxy=(local or "").strip())

        def save_net_cfg(mode, local, remote):
            CFG.net = net_of(mode, local, remote)
            save_net(CFG.net)
            return "✅ 已保存 → hosts.json（权限 600）"

        def test_net(host_name, mode, local, remote):
            try:
                host = CFG.get_host(host_name)
            except ValueError as ex:
                return f"⚠ {ex}"
            CFG.net = net_of(mode, local, remote)
            ok, text = runner.test_proxy(CFG, host)
            return ("✅ " if ok else "⚠ ") + text

        # -------------------------------------------------- 状态探测
        def do_refresh(host_name, qf_cur, kv_cur):
            """探测机器 + 刷新 qformat/KV 词表（保持原返回形状）。"""
            try:
                host = CFG.get_host(host_name)
            except ValueError as ex:
                return (f"⚠ {ex}", gr.update(), qf_cur, gr.update(), kv_cur)
            st = runner.probe_host(host, app_cfg=CFG)
            md = f"**{host.identity}**\n"
            if st.error:
                md += f"⚠ {st.error}\n\n"
            md += " | ".join(f"{k}: {v}" for k, v in st.as_dict().items())
            qf = {"model": vocab.all_qformats(), "kv": vocab.KV_QFORMATS,
                  "source": "内置（未连上机器）"}
            if st.reachable:
                try:
                    qf = vocab.fetch_remote_qformats(host)
                except Exception as ex:
                    qf["source"] = f"内置（读机器词表失败: {ex}）"
            choices = vocab.render_qformat_dropdown(qf["model"])
            kvs = qf.get("kv") or vocab.KV_QFORMATS
            kv_up = gr.update(choices=kvs)
            if kv_cur not in kvs:
                kv_cur, kv_up = (kvs[0] if kvs else "none"), gr.update(choices=kvs,
                                                                       value=kvs[0])
            names = [v for _, v in choices]
            qf_up = gr.update(choices=choices)
            if qf_cur not in names:
                qf_cur = names[0] if names else qf_cur
                qf_up = gr.update(choices=choices, value=qf_cur)
            md += f"\n\n词表来源: {qf['source']}"
            return (md, qf_up, qf_cur, kv_up, kv_cur)

        def initial_probe():
            if not CFG.host_names():
                return (gr.update(), gr.update(), "（还没配机器）", "nvfp4",
                        gr.update(), gr.update(), "none")
            md, qf_up, qf_v, kv_up, kv_v = do_refresh(CFG.host_names()[0], "nvfp4", "none")
            return (gr.update(value=_mode_line(CFG.host_names()[0])),
                    gr.update(value=_net_line(CFG.host_names()[0])), md, qf_v,
                    qf_up, kv_up, kv_v)

        # -------------------------------------------------- 命令预览
        def refresh_preview(host_name, *a):
            try:
                cmd = build_remote_cmd(CFG.get_host(host_name), build_params(list(a)))
                return _prettify_cmd(cmd)
            except Exception as ex:
                return f"（拼参出错: {ex}）"

        def sizes_rows(names):
            return [[1024] for _ in (names or [])] if names else None

        # -------------------------------------------------- 量化任务
        def do_start(host_name, *a):
            try:
                p = build_params(list(a))
                host = CFG.get_host(host_name)
            except Exception as ex:
                yield (f"⚠ {ex}", _prog_html(progress.ProgressReport(
                    error=str(ex))), "参数有误", None)
                return
            cmd = build_remote_cmd(host, p)
            st = runner.probe_host(host)
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
                why = check_local_model(host, model)
                if why:
                    yield ("⚠ 模型路径有问题", _prog_html(progress.ProgressReport(error=why)),
                           why, None)
                    return
            gpu_warn = ""
            if st.gpu_busy:
                gpu_warn = (f"⚠ GPU 当前 {st.gpu_util}% 占用中：不抢占现有任务，"
                            f"会和它抢显存。确认那是无用进程再启动（「■ 中止」能停自己的）。\n")
            note = gpu_warn + f"执行位置：{host.where}\n"
            yield note, _prog_html(progress.ProgressReport(stage_idx=0,
                                                           stage="提交任务")), "任务提交中…", None
            try:
                task = runner.start_task(host, host.task_dir,
                                         task_slug(model),
                                         str(p.get("export_path") or ""),
                                         cmd, runner.proxy_for(CFG, host), lambda s: None)
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
        def do_venv(host_name):
            try:
                host = CFG.get_host(host_name)
            except ValueError as ex:
                return f"⚠ {ex}"
            buf: list[str] = []
            try:
                runner.ensure_venv(host, lambda s: buf.append(s), app_cfg=CFG)
            except Exception as ex:
                return f"⚠ 创建失败：{ex}"
            text = "\n".join(buf) or "（无输出）"
            rep = progress.parse_install(buf)
            head = "✅ venv 就绪" if not rep.error else f"⚠ {rep.error}"
            return f"{head} · {rep.detail}\n```\n{text[-4000:]}\n```"

        def do_eval_env(host_name):
            try:
                host = CFG.get_host(host_name)
            except ValueError as ex:
                return f"⚠ {ex}"
            buf: list[str] = []
            try:
                runner.ensure_eval_env(host, lambda s: buf.append(s), app_cfg=CFG)
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
            try:
                hint = state.remote_cmd.split("--export_path")[-1].split()[0].strip("'\"")
            except Exception:
                pass
            mine = validate.fetch_remote_export(state.host, log, export_dir_hint=hint)
            if mine.error:
                return f"⚠ {mine.error}", None, "（无数据）", ""
            md = (f"**{mine.source}**\nquant_method: {mine.quant_method}\n"
                  f"ignore: {validate.ignore_preview(mine)}")
            diff = ""
            if official.strip():
                try:
                    off = validate.fetch_official_summary(
                        official, proxy=CFG.net.local_proxy or "")
                    diff = "\n\n".join(validate.diff_summaries(mine, off))
                except Exception as ex:
                    diff = f"⚠ 官方对照失败: {ex}"
            files = ""
            if mine.files:
                files = f"\n\n**导出目录文件**\n```\n{mine.files}\n```"
            return md, validate.to_table_rows(mine), diff, files

        # -------------------------------------------------- 评测
        def do_fetch_baseline(host_name, quant_path):
            q = (quant_path or "").strip()
            if not q:
                return "（先填量化模型路径）", gr.update()
            try:
                host = CFG.get_host(host_name)
            except ValueError as ex:
                return f"⚠ {ex}", gr.update()
            r, out, _ = runner.run_remote(host, f"cat {q_remote(q, host)}/config.json "
                                                f"2>/dev/null", timeout=20)
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

        def do_eval(host_name, quant, baseline, tp, port, batch, tasks):
            try:
                host = CFG.get_host(host_name)
            except Exception as ex:
                yield f"⚠ {ex}", _prog_html(progress.ProgressReport(error=str(ex))), \
                    "参数有误", None
                return
            ok, msg = runner.probe_eval_env(host)
            if not ok:
                yield (f"⚠ {msg}", _prog_html(progress.ProgressReport(error=msg)),
                       "请先点上方「一键装评测环境」（vllm + lm_eval）", None)
                return
            note = f"执行位置：{host.where}\n"
            yield note, _prog_idle("启动评测"), "评测启动中…", None
            try:
                script = build_eval_script(host, quant, baseline, tasks,
                                           int(tp or 1), int(port or 8000),
                                           str(batch or "auto"))
                ts = time.strftime("%Y%m%d-%H%M%S")
                tdir = f"{host.task_dir.rstrip('/')}/eval_{ts}"
                task = runner.start_script_task(host, tdir, script,
                                                runner.proxy_for(CFG, host),
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

        def do_fetch_results(host_name):
            try:
                host = CFG.get_host(host_name)
            except Exception as ex:
                return f"⚠ {ex}", None
            results = runner.fetch_eval_results(host, host.task_dir)
            if not results:
                return "（没找到 results_*.json）", None
            rows, note = _result_tables(results)
            return f"**{len(results)} 个结果文件**\n{note}", rows

        # ============================================================ 绑定
        for inp in all_inputs[1:]:
            inp.change(refresh_preview, all_inputs, preview_md)
        combo_dd.change(lambda c: list(vocab.COMBOS.get(c, [])) if c else [],
                        combo_dd, ds_dd)
        sizes_dd.change(sizes_rows, sizes_dd, sizes_table)

        refresh_btn.click(do_refresh, [host_dd, qf_val_st, kv_val_st],
                          [status_md, qf_dd, qf_val_st, kv_dd, kv_val_st])
        host_dd.change(do_refresh, [host_dd, qf_val_st, kv_val_st],
                       [status_md, qf_dd, qf_val_st, kv_dd, kv_val_st])
        host_dd.change(mode_line, host_dd, mode_md)
        host_dd.change(lambda h: gr.update(value=h), host_dd, pick_dd)
        host_dd.change(refresh_net, host_dd, net_md)
        ds_dd.change(lambda d: d or [], ds_dd, sizes_dd)

        start_btn.click(do_start, all_inputs, [proxy_note, prog_html, log_box, task_state])
        abort_btn.click(do_abort, task_state, proxy_note)
        validate_btn.click(do_validate, [task_state, official_in],
                           [result_md, group_table, diff_md, files_md])
        venv_btn.click(do_venv, host_dd, venv_log)
        eval_env_btn.click(do_eval_env, host_dd, venv_log)

        fetch_b_btn.click(do_fetch_baseline, [host_dd, ev_q_in], [fetch_b_note, ev_b_in])
        ev_preset.change(do_ev_preset, ev_preset, ev_tasks)
        ev_run.click(do_eval, [host_dd, ev_q_in, ev_b_in, ev_tp, ev_port, ev_batch,
                               ev_tasks], [ev_note, ev_prog, ev_log, eval_state])
        ev_abort.click(do_ev_abort, eval_state, ev_note)
        ev_fetch.click(do_fetch_results, host_dd, [ev_result_md, ev_table])

        # 连接设置
        pick_dd.change(pick_host, [pick_dd, hosts_st],
                       [ed_name, ed_mode, ed_ssh, ed_port, ed_pass, ed_repo, ed_venv,
                        ed_task, ed_cache, save_host_btn])
        new_btn.click(new_host, None,
                      [ed_name, ed_mode, ed_ssh, ed_port, ed_pass, ed_repo, ed_venv,
                       ed_task, ed_cache, save_host_btn])
        ed_mode.change(toggle_ssh, ed_mode, [ed_ssh, ed_port, ed_pass])
        save_host_btn.click(
            save_host,
            [pick_dd, ed_name, ed_mode, ed_ssh, ed_port, ed_pass, ed_repo, ed_venv,
             ed_task, ed_cache, hosts_st],
            [host_dd, pick_dd, host_list_md, host_note, hosts_st])
        del_btn.click(del_host, [pick_dd, hosts_st],
                      [host_dd, pick_dd, host_list_md, host_note, hosts_st])
        test_btn.click(test_conn, pick_dd, host_note)
        recon_btn.click(reconnect, host_dd, conn_note)
        net_mode.change(lambda m: (gr.update(interactive=m != PROXY_DIRECT),
                                   gr.update(interactive=m == PROXY_MANUAL or
                                             m == PROXY_AUTO)),
                        net_mode, [net_local, net_remote])
        save_net_btn.click(save_net_cfg, [net_mode, net_local, net_remote], net_note)
        test_net_btn.click(test_net, [host_dd, net_mode, net_local, net_remote], net_note)

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
    return gr.themes.Ocean(
        primary_hue="emerald", secondary_hue="sky",
        font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
    )


if __name__ == "__main__":
    main()

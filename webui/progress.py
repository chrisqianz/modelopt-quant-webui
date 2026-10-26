"""把任务日志尾部翻成「现在在干什么 + 干了多少」。

hf_ptq.py 本身几乎不打阶段大字，真正的进度来自 tqdm —— tqdm 写 stderr，
`2>&1` 落进日志文件后每条进度以 \\r 分隔留在文件里，所以按 \\r 切片取最后
一段就是当前进度。阶段则靠 hf_ptq.py 里确实存在的 print 文本来定位。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 阶段顺序 = hf_ptq.py 的实际执行顺序；编号用于画「第几步」
STAGES = [
    "启动",
    "下载模型权重 (HuggingFace)",
    "加载模型到显存",
    "准备校准数据",
    "校准并量化",
    "导出量化 checkpoint",
    "完成",
]

# (匹配文本, 阶段下标)；取日志里命中的最大下标（阶段单调推进）
_MARKERS: list[tuple[str, int]] = [
    ("loading pretrained model", 2),
    ("loading checkpoint shards", 2),
    ("image-text calibration enabled", 3),
    ("use calib batch_size", 3),
    # datasets 库自己打的进度行。hf_ptq 那句 "Use calib batch_size" 会被后面几万条
    # \r 进度块冲出日志尾部，只靠它认不出「已经在弄校准数据」了
    ("downloading readme", 3),
    ("downloading metadata", 3),
    ("resolving data files", 3),
    ("downloading data files", 3),
    ("extracting data files", 3),
    ("generating train split", 3),
    ("generating validation split", 3),
    ("use recipe", 4),
    ("updating full_model with quantized", 5),
    ("layerwise export", 5),
    ("no quantization applied, export", 5),
    ("kv cache quantization", 4),
    ("example test input:", 6),
]

_DONE_MARKERS = (
    "quantized model exported to:",
    "quantized speculative decoding checkpoint exported to:",
)

# 这些文字一出现就是死了，别再显示「进行中」骗人
_FATAL: list[tuple[str, str]] = [
    ("cuda out of memory", "显存不够（CUDA OOM）：调小 batch_size / 开 --enable_max_memory 或换卡"),
    ("out of memory error", "显存不够（OOM）"),
    ("no space left on device", "磁盘满了"),
    ("we couldn't connect to huggingface", "连不上 huggingface.co：代理没生效，检查「网络与代理」"),
    ("maximum number of retries exceeded", "HF 重试次数用完：网络不通或被限流"),
    ("401 client error", "HF 401：私有模型需要 token（HF_TOKEN 或 hf auth login）"),
    ("403 client error", "HF 403：无权限访问该仓库"),
    ("gated dataset", "HF 受限数据集（gated）：先在网页同意数据集条款，再在 GPU 机 hf auth login 或设 HF_TOKEN 重跑"),
    ("repository not found", "HF 仓库不存在或名字写错"),
    ("traceback (most recent call last)", "进程抛异常退出了"),
    ("killed", "进程被系统杀掉（多半是内存/显存不足）"),
]

# tqdm 一行： 前缀:  42%|████ | 1.23G/2.94G [00:31<00:44, 38.5MB/s]
# cur/tot 写成可选：`0%| | 0/5 [00:00<?, ?it/s]` 和没计数的进度条也得能命中
_RE_TQDM = re.compile(
    r"(?P<pre>[^\r\n|%]{0,90}?)?[ \t]*(?P<pct>\d{1,3})%\|\s*(?P<bar>[^|]*)\|\s*"
    r"(?:(?P<cur>[\d.]+[KMGTP]?)\s*/\s*(?P<tot>[\d.]+[KMGTP]?)\s*)?"
    r"(?:\[(?P<stat>[^\]]*)\])?"
)
# uv/pip： Downloading 246 packages / [12/246] ...
_RE_COUNT = re.compile(r"(?P<cur>\d+)\s*/\s*(?P<tot>\d+)(?![\d.])")
_RE_SPEED = re.compile(r"(?P<v>[\d.]+)\s*(?P<u>[KMG]?i?B?/s|it/s|s/it)")
_RE_LEFT = re.compile(r"<\s*(?P<left>[\d:]+)")

# 下载中的文件名长这样，用来判断「这条进度条是下载还是算得慢」
_DOWNLOAD_HINT = (".safetensors", ".bin", ".gguf", ".npz", ".pt", "fetching",
                  "downloading", "model-", "tokenizer", "config.json", "pytorch_model")

# datasets 加载校准数据的进度条前缀。必须先于 _DOWNLOAD_HINT 判：它前缀里也带
# "downloading"，否则「Downloading data files」会被当成在下权重，阶段永远卡在第二步
_DATASET_HINT = ("downloading data", "downloading readme", "downloading metadata",
                 "extracting data", "generating train split", "generating validation split",
                 "generating test split", "resolving data files", "loading cached",
                 "computing statistics")
# tqdm 的 desc 被完整用作前缀的那些（没有更具体的文字可匹）
_DATASET_PREFIX = {"map", "tokenizing", "filter", "filterfn", "shuffle", "cast",
                   "sort", "gather", "encode", "preprocess"}


@dataclass
class ProgressReport:
    stage_idx: int = 0
    stage: str = STAGES[0]
    pct: float | None = None      # 0-100，None = 这一阶段没百分比可报
    bar: str = ""                 # 画好的文本进度条
    detail: str = ""              # 1.23G / 2.94G · 38.5MB/s · 剩余 00:44
    item: str = ""                # 当前文件 / 当前批次
    done: bool = False
    error: str = ""

    @property
    def active(self) -> bool:
        return not self.done and not self.error

    def markdown(self) -> str:
        """给前端状态卡用的一小块 Markdown。"""
        if self.error:
            return (f"❌ **{self.stage}** — 出错了\n\n"
                    f"> {self.error}\n\n`{self.item}`" if self.item else
                    f"❌ **{self.stage}** — 出错了\n\n> {self.error}")
        if self.done:
            return f"✅ **{STAGES[-1]}** — 量化流程跑完了，可以看结果了"
        head = f"**第 {self.stage_idx + 1}/{len(STAGES)} 步 · {self.stage}**"
        if self.pct is None:
            return f"⏳ {head}\n\n`{self.detail or self.item}`" if (self.detail or self.item) \
                else f"⏳ {head}"
        line = f"{head}\n\n`{self.bar}` **{self.pct:.0f}%**"
        if self.detail:
            line += f"\n\n{self.detail}"
        return line


def _bar(pct: float, width: int = 24) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


_RE_BYTES = re.compile(r"^\d+(?:\.\d+)?[KMGTP]$", re.I)


def _amount(cur: str | None, tot: str | None) -> str:
    # `1.23G/2.94G` 是字节，`47/100` 是步数 —— 别给后者脑补单位
    if not cur or not tot:
        return ""
    if _RE_BYTES.match(cur) and _RE_BYTES.match(tot):
        return f"{cur} / {tot}"
    return f"{cur} / {tot} 步"


def _tqdm_chunk(text: str) -> str:
    """日志里最后一条还在跳的进度条（tqdm 用 \\r 覆盖同一行）。"""
    for chunk in reversed(text.split("\r")[-60:]):
        if _RE_TQDM.search(chunk):
            return chunk
    return ""


def _pick_stage(lines: list[str], downloading: bool, dataset: bool = False) -> int:
    # 数据集进度条说明已经在准备校准数据了，比「在下权重」更靠后
    idx = 3 if dataset else (1 if downloading else 0)
    for ln in lines:
        low = ln.lower().strip()
        for key, target in _MARKERS:
            if key in low:
                idx = max(idx, target)
    return min(idx, len(STAGES) - 1)


def _scan_error(text: str) -> str:
    low = text.lower()
    for key, msg in _FATAL:
        if key in low:
            return msg
    return ""


def parse(lines_or_text, extra_note: str = "") -> ProgressReport:
    """解析任务日志尾部。extra_note 用来塞「任务还没输出」这类外部信息。"""
    text = lines_or_text if isinstance(lines_or_text, str) else "\n".join(lines_or_text)
    rep = ProgressReport()
    if not text.strip():
        rep.detail = extra_note or "任务还没有输出…"
        return rep

    low = text.lower()
    if any(m in low for m in _DONE_MARKERS):
        rep.done = True
        rep.stage_idx = len(STAGES) - 1
        rep.stage = STAGES[-1]
        rep.pct, rep.bar = 100.0, _bar(100)
        return rep

    err = _scan_error(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # 进度条：只有最后一段才是当前值
    m = _RE_TQDM.search(_tqdm_chunk(text))
    downloading = False
    dataset = False
    if m:
        pre = (m.group("pre") or "").strip().rstrip(":").strip()
        pct = max(0.0, min(100.0, float(m.group("pct"))))
        pre_low = pre.lower()
        dataset = pre_low in _DATASET_PREFIX or any(h in pre_low for h in _DATASET_HINT)
        downloading = not dataset and any(h in pre_low for h in _DOWNLOAD_HINT)
        rep.pct, rep.bar, rep.item = pct, _bar(pct), pre[:70]
        bits = [a for a in [_amount(m.group("cur"), m.group("tot"))] if a]
        stat = m.group("stat") or ""
        sp = _RE_SPEED.search(stat)
        if sp:
            bits.append(f"{sp.group('v')} {sp.group('u')}")
        left = _RE_LEFT.search(stat)
        if left:
            bits.append(f"剩余 {left.group('left')}")
        rep.detail = " · ".join(bits)

    rep.stage_idx = _pick_stage(lines, downloading, dataset)
    rep.stage = STAGES[rep.stage_idx]

    # 没有百分比的阶段，退化成「最后一行在忙什么」
    if rep.pct is None:
        for ln in reversed(lines):
            s = ln.strip()
            if s and "%" not in s[:3] and not s.startswith("#"):
                rep.item = s[:90]
                break
    if rep.stage_idx >= 4 and rep.pct is None:
        rep.detail = "校准/量化按层推进，hf_ptq 不给百分比，看下面日志滚动"
    elif rep.stage_idx == 3 and rep.pct is None:
        # 数据集已在 HF 缓存里时 datasets 不打进度条，别让卡片看着像卡死
        rep.detail = "校准数据下载 / 切分中；数据集若已在缓存里就不会有进度条"
    elif dataset and rep.pct is not None:
        rep.detail = (rep.detail + " · 校准数据集").lstrip(" ·")

    if err:
        rep.error = err
    return rep


def parse_install(lines_or_text) -> ProgressReport:
    """uv / pip 安装日志（建 venv、装评测环境用）。"""
    text = lines_or_text if isinstance(lines_or_text, str) else "\n".join(lines_or_text)
    rep = ProgressReport(stage="安装环境", stage_idx=1)
    if not text.strip():
        rep.detail = "还没有输出…"
        return rep
    tail = text.splitlines()[-1:] or [""]
    rep.item = tail[0].strip()[:90]
    last = ""
    for chunk in reversed(text.split("\r")[-60:]):
        if _RE_COUNT.search(chunk):
            last = chunk
            break
    m = _RE_COUNT.search(last) if last else None
    if m and "%" not in last:
        cur, tot = int(m.group("cur")), int(m.group("tot"))
        if tot > 3 and cur <= tot:
            rep.pct = cur / tot * 100
            rep.bar = _bar(rep.pct)
            rep.detail = f"{cur} / {tot} 个包"
    for key, msg in _FATAL:
        if key in text.lower():
            rep.error = msg
            break
    return rep

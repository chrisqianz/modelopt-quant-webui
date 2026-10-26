"""本机执行器：状态探测 / venv / 代理 / 任务生命周期。

所有操作都是只读探测或用户显式发起的任务；GPU 占用不抢占、不改动。
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import progress
from .config import (
    PROXY_AUTO,
    PROXY_DIRECT,
    PROXY_MANUAL,
    AppConfig,
    NetConfig,
    resolve_relative,
)

_log = logging.getLogger(__name__)


def q_remote(path: str | Path) -> str:
    """展开 ~ / 项目内相对路径再转义。存在理由：shlex.quote 会把 ~ 括进单引号，shell 不展开。

    只做转义，不做路径归一化：resolve_relative 返回 Path，会吃掉结尾的 `/`。
    用户在 UI 里写的 `~/recipes/dir/` 就该原样送进命令 —— 尾斜杠是否保留不该由
    一个转义函数悄悄决定（golden 快照因此能保持与重构前逐字一致）。
    """
    raw = str(path)
    out = str(resolve_relative(raw).expanduser())
    if raw.endswith("/") and not out.endswith("/"):
        out += "/"
    return shlex.quote(out)

# ---------------------------------------------------------------- 单次执行

def _exec(argv: list[str], timeout: int = 60,
          env: dict | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"命令超时 ({timeout}s): {' '.join(argv[-2:])[:120]}"


def run_shell(cmd: str, timeout: int = 60,
              env_extra: dict | None = None) -> tuple[int, str, str]:
    """在本机跑一段 bash。返回 (rc, stdout, stderr)。

    env_extra：额外注入子进程的环境变量（如 HF_TOKEN）。走 subprocess env
    而非命令串：令牌不进 ps 可见的命令行，也不落盘 run.sh。"""
    env = {**os.environ, **env_extra} if env_extra else None
    return _exec(["bash", "-c", cmd], timeout=timeout, env=env)


def hf_task_env(cfg: AppConfig) -> dict | None:
    """任务启动用的 HF 环境（令牌未配则 None，行为与从前一致）。"""
    tok = getattr(cfg.net, "hf_token", "")
    return {"HF_TOKEN": tok} if tok else None


def check_connection() -> str:
    rc, out, err = run_shell("echo __OK__; hostname; echo HOME=$HOME")
    if rc != 0 or "__OK__" not in out:
        return f"✗ 本机 shell 不可用：{(err or out)[:200]}"
    body = out.replace("__OK__", "").strip()
    return f"✓ 本机执行可用 · {body}"


# ---------------------------------------------------------------- 状态探测

@dataclass
class HostStatus:
    reachable: bool = False
    gpu: str = ""
    gpu_util: int | None = None
    gpu_mem: str = ""
    gpu_busy: bool = False
    repo_ok: bool = False
    venv_ok: bool = False
    uv_ok: bool = False
    disk_free: str = ""
    hf_cache: str = ""
    hf_direct_ok: bool = False      # 本机直连 huggingface.co 是否可达
    hf_proxy_ok: bool = False       # 按当前代理配置能否访问 HF
    proxy_note: str = ""            # 实际会用哪条路下载 HF
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "可达": "✓" if self.reachable else "✗",
            "GPU": self.gpu,
            "GPU 利用率": f"{self.gpu_util}%" if self.gpu_util is not None else "—",
            "GPU 显存": self.gpu_mem or "—",
            "GPU 状态": ("⚠ 占用中（任务仍可排队提交，但建议等空闲）" if self.gpu_busy
                        else "空闲" if self.gpu_util is not None else "—"),
            "uv": "✓" if self.uv_ok else "✗",
            "ModelOpt 仓库": f"✓ {self.repo_ok}" if self.repo_ok else "✗ 路径不存在",
            "venv": "✓ 已创建" if self.venv_ok else "✗ 未创建（点状态栏按钮创建）",
            "磁盘可用": self.disk_free or "—",
            "HF 缓存": self.hf_cache or "—",
            "HF 网络": _hf_status_text(self),
        }

def _hf_status_text(st: HostStatus) -> str:
    if st.hf_direct_ok:
        return "✓ 直连可达"
    if st.hf_proxy_ok:
        return f"✓ 直连不通，走代理可达（{st.proxy_note}）"
    return f"⚠ 直连不通，代理未就绪（{st.proxy_note or '未配置'}）"

def probe_hf(proxy: str = "") -> tuple[bool, str]:
    """探测本机能否访问 huggingface.co。proxy 传 http:// 或 socks5h:// 地址，空 = 直连。
    返回 (可达, 说明)。"""
    env = _proxy_env(proxy) if proxy else ""
    cmd = (f"{env}curl -s -o /dev/null -m 15 -w '%{{http_code}} %{{time_total}}' "
           f"https://huggingface.co/api/models")
    r, out, err = run_shell(cmd, timeout=30)
    if r != 0:
        return False, f"✗ 探测失败：{(err or out)[:200]}"
    code, _, secs = out.partition(" ")
    detail = f"HTTP {code}"
    with contextlib.suppress(ValueError):
        detail += f"，{float(secs):.1f}s"
    if code != "200" and err:
        detail += f"（{err.splitlines()[-1][:80]}）"
    return code == "200", detail

_HF_OK: tuple[bool, float] | None = None     # (直连可达, 缓存过期时刻)


def hf_direct_ok(ttl: float = 300.0) -> bool:
    """本机直连可达性（探一次缓存 ttl 秒；每次刷状态都真探一遍太慢）。"""
    global _HF_OK
    if _HF_OK and _HF_OK[1] > time.monotonic():
        return _HF_OK[0]
    ok, _ = probe_hf()
    _HF_OK = (ok, time.monotonic() + ttl)
    return ok

def probe_host(cfg: AppConfig) -> HostStatus:
    st = HostStatus()
    msg = check_connection()
    if not msg.startswith("✓"):
        st.error = msg
        return st
    st.reachable = True

    # GPU（只读 nvidia-smi，不产生任何负载）
    r, out, _ = run_shell(r"nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1", timeout=15)
    if r == 0 and out:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            st.gpu, st.gpu_mem = parts[0], f"{re.sub(r'[^0-9]',' ',parts[2]).strip()}/{re.sub(r'[^0-9]',' ',parts[3]).strip()} MiB"
            st.gpu_util = int(re.sub(r"[^0-9]", "", parts[1]) or 0)
            st.gpu_busy = st.gpu_util >= 80

    # 环境
    r, out, _ = run_shell("command -v uv || ls ~/.local/bin/uv 2>/dev/null", timeout=10)
    st.uv_ok = r == 0 and bool(out)

    r, out, _ = run_shell(f"test -d {q_remote(cfg.local.repo)}/examples/hf_ptq && echo ok", timeout=10)
    st.repo_ok = out == "ok"

    r, out, _ = run_shell(f"test -x {q_remote(cfg.local.venv)}/bin/python && echo ok", timeout=10)
    st.venv_ok = out == "ok"

    r, out, _ = run_shell(f"df -h {q_remote(cfg.local.hf_cache)} | awk 'NR==2{{print $4}}'", timeout=10)
    st.disk_free = out

    r, out, _ = run_shell(f"du -sh {q_remote(cfg.local.hf_cache)} 2>/dev/null | awk '{{print $1}}'", timeout=20)
    st.hf_cache = out

    # HF 可达性：国内直连常不通，按网络面板的模式逐条验证
    st.hf_direct_ok = hf_direct_ok()
    net: NetConfig = cfg.net
    if st.hf_direct_ok:
        st.proxy_note = "直连"
    elif net.mode == PROXY_DIRECT:
        st.proxy_note = "已设为「不使用代理」"
    elif net.mode == PROXY_MANUAL:
        url = normalize_proxy(net.proxy)
        if url:
            st.hf_proxy_ok, st.proxy_note = probe_hf(proxy=url)
        else:
            st.proxy_note = "未填代理地址（设置里填）"
    elif net.mode == PROXY_AUTO:
        url = normalize_proxy(net.proxy)
        if url:
            st.hf_proxy_ok, detail = probe_hf(proxy=url)
            st.proxy_note = f"本机代理 {detail}"
        else:
            st.proxy_note = "未配置代理地址（设置里填）"
    return st

# ---------------------------------------------------------------- 代理

def normalize_proxy(url: str) -> str:
    """把用户手填的代理地址成能直接注进 env 的形式：
    socks5:// → socks5h://（让代理端解 DNS，本机常解不了 huggingface.co）；
    没写 scheme 的（用户常直填 127.0.0.1:7890）补 http://，requests 不认无 scheme 的代理。"""
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if u.startswith("socks5://"):
        return "socks5h://" + u[len("socks5://"):]
    if "://" not in u:
        return "http://" + u
    return u

def _parse_proxy_addr(url: str) -> tuple[str, str, str] | None:
    """'socks5://127.0.0.1:7890' / '127.0.0.1:7890' → (scheme, host, port)。"""
    m = re.match(r"^\s*(?:(\w+)://)?([^:/@\s]+):(\d{1,5})/?\s*$", url or "")
    if not m:
        return None
    return (m.group(1) or "socks5").lower(), m.group(2), m.group(3)

def proxy_url_for(cfg: AppConfig) -> tuple[str, str]:
    """→ (proxy url, 说明)。只有本机代理一条路，地址一律手填。"""
    if cfg.net.mode == PROXY_DIRECT:
        return "", "按设置不用代理"
    url = normalize_proxy(cfg.net.proxy)
    if not url:
        return "", "没填代理地址"
    if cfg.net.mode == PROXY_MANUAL:
        return url, f"强制走 {url}"
    if hf_direct_ok():
        return "", "直连可用，没用代理"
    return url, f"直连不通，走 {url}"


def proxy_for(cfg: AppConfig) -> str:
    """按网络面板的配置，返回要注入执行环境（nohup 命令前缀）的 proxy 变量串。"""
    url, _ = proxy_url_for(cfg)
    return _proxy_env(url) if url else ""


def test_proxy(cfg: AppConfig, url: str = "") -> tuple[bool, str]:
    """「测试网络」：按当前配置（或 url 指定的地址）从执行侧实打一次 huggingface.co。"""
    net: NetConfig = cfg.net
    url = normalize_proxy(url)
    if url:
        return probe_hf(proxy=url)
    if net.mode == PROXY_DIRECT:
        ok, detail = probe_hf()
        return ok, ("直连可达" if ok else f"直连不通：{detail}")
    if net.mode == PROXY_MANUAL:
        url = normalize_proxy(net.proxy)
        if not url:
            return False, "代理地址是空的，先填一个这台机器连得到的地址"
        ok, detail = probe_hf(proxy=url)
        return ok, (f"{url} 可用（{detail}）" if ok else f"{url} 不通：{detail}")
    if hf_direct_ok():
        return True, "直连 huggingface.co 可达，不需要代理"
    use, note = proxy_url_for(cfg)
    if not use:
        return False, f"直连不通且无代理可用：{note}"
    ok, detail = probe_hf(proxy=use)
    return ok, (f"{note} · 实打成功（{detail}）" if ok
                else f"{note} · 但 HF 取不到：{detail}")

def _proxy_env(proxy: str) -> str:
    p = shlex.quote(normalize_proxy(proxy))
    return (f"export http_proxy={p} https_proxy={p} "
            f"HF_HUB_DISABLE_IMPLICIT_TOKEN=1 no_proxy=${{no_proxy:+$no_proxy,}}localhost,127.0.0.1;")

# ---------------------------------------------------------------- venv

def _resolve_uv() -> str:
    r, out, _ = run_shell("command -v uv || echo ~/.local/bin/uv", timeout=10)
    return out.splitlines()[-1] if r == 0 and out else "~/.local/bin/uv"

def venv_setup_cmd(uv: str, venv: str, repo: str, proxy_env: str) -> str:
    """ensure_venv 的一次性安装脚本（纯函数，供单测）。

    主安装 modelopt[hf] 的退出码决定 VENV_DONE rc；尾部两个可选 GPU kernel 是
    best-effort，成败都不影响主流程 —— 缺了只是线性注意力走参考实现、校准慢，
    正确性不受影响（transformers 自己的回退就是数学等价的）。"
    """
    uvpip = f"VIRTUAL_ENV={venv} {uv} pip"
    return (
        f"mkdir -p {venv} && "
        f"{proxy_env}"
        f"{uv} venv {venv} --python 3.12 && "
        f"cd {repo} && "
        f"{uvpip} install -e '.[hf]' 2>&1; rc=$?; "
        # 线性注意力快速核（Qwen3.5 等混合架构）：--no-deps 防依赖解析动钉死的
        # torch cu130；import 失败即回滚，不给 transformers 留半残的可用性探测
        f"{uvpip} install --no-deps flash-linear-attention 2>&1 | tail -1; "
        f"{venv}/bin/python -c 'import fla' 2>/dev/null "
        f"|| {uvpip} uninstall -y flash-linear-attention >/dev/null 2>&1; "
        # 只吃预编译轮子：新平台（如 sm_120+CUDA13）无 wheel 时秒失败，绝不源码编译
        f"{uvpip} install --only-binary :all: causal-conv1d 2>&1 | tail -1; "
        f"echo VENV_DONE rc=$rc"
    )

def ensure_venv(log, app_cfg: AppConfig) -> None:
    """在本机创建 venv 并安装 modelopt[hf]。长耗时操作走 nohup，轮询日志。
    app_cfg 提供时顺带注入 HF 代理（国内 pypi/HF 都可能受限，统一走代理更稳）。"""
    venv = q_remote(app_cfg.local.venv)
    r, out, _ = run_shell(f"test -x {venv}/bin/python && echo ok", timeout=10)
    if out == "ok":
        log("venv 已存在，跳过创建")
        return
    uv = _resolve_uv()
    repo = q_remote(app_cfg.local.repo)
    proxy_env = proxy_for(app_cfg)
    cmd = venv_setup_cmd(uv, venv, repo, proxy_env)
    fd, log_path = tempfile.mkstemp(prefix="modelopt_venv_setup_", suffix=".log")
    os.close(fd)
    bg_cmd = f"nohup bash -c {shlex.quote(cmd)} > {shlex.quote(log_path)} 2>&1 & echo $!"
    r, out, err = run_shell(bg_cmd, timeout=30)
    if r != 0:
        log(f"venv 创建启动失败: {err}")
        return
    log("venv 创建中（uv venv + pip install -e '.[hf]'，通常 5-15 分钟；"
        + ("已走代理" if proxy_env else "直连安装") + "）…")
    _wait_for_marker(log_path, "VENV_DONE", log, wait_s=15, msg="…仍在安装依赖",
                     parser=progress.parse_install)

def ensure_eval_env(log, app_cfg: AppConfig) -> None:
    """在已有 venv 里加装评测栈：vllm + lm_eval。长耗时操作走 nohup，轮询日志。"""
    venv = q_remote(app_cfg.local.venv)
    r, out, _ = run_shell(f"test -x {venv}/bin/python && echo ok", timeout=10)
    if out != "ok":
        log("⚠ 基础 venv 不存在，请先点状态栏的「一键创建 venv」")
        return
    uv = _resolve_uv()
    proxy_env = proxy_for(app_cfg)
    cmd = (
        f"{proxy_env}"
        f"VIRTUAL_ENV={venv} {uv} pip install 'vllm>=0.10.1' 'lm_eval[ifeval]>=0.4.12,<0.5' 2>&1; "
        f"echo EVAL_ENV_DONE rc=$?"
    )
    fd, log_path = tempfile.mkstemp(prefix="modelopt_eval_env_", suffix=".log")
    os.close(fd)
    bg_cmd = f"nohup bash -c {shlex.quote(cmd)} > {shlex.quote(log_path)} 2>&1 & echo $!"
    r, out, err = run_shell(bg_cmd, timeout=30)
    if r != 0:
        log(f"评测环境安装启动失败: {err}")
        return
    log("评测环境安装中（vllm + lm_eval，下载体积大，通常 10-30 分钟；"
        + ("已走代理" if proxy_env else "直连安装") + "）…")
    _wait_for_marker(log_path, "EVAL_ENV_DONE", log, wait_s=20, msg="…仍在安装依赖",
                     parser=progress.parse_install)

def _wait_for_marker(log_path: str, marker: str, log, wait_s: int,
                     msg: str, parser=None) -> None:
    while True:
        r, out, _ = run_shell(f"tail -c 8000 {shlex.quote(log_path)} 2>/dev/null",
                              timeout=15)
        if marker in (out or ""):
            log("完成" if "rc=0" in out else f"失败（见 {log_path}）: {out}")
            return
        time.sleep(wait_s)
        rep = parser((out or "").splitlines()) if parser else None
        log(rep.markdown if rep and rep.markdown else msg)

# ---------------------------------------------------------------- 评测

def probe_eval_env(cfg: AppConfig) -> tuple[bool, str]:
    """探测本机 venv 里是否已装 vllm + lm_eval（只读，不占 GPU）。"""
    venv = q_remote(cfg.local.venv)
    r, out, _ = run_shell(f"test -x {venv}/bin/python && echo ok", timeout=10)
    if out != "ok":
        return False, "基础 venv 未创建（先点状态栏「一键创建 venv」）"
    r, out, _ = run_shell(
        f"{venv}/bin/python -c 'import vllm, lm_eval; print(vllm.__version__)' 2>/dev/null",
        timeout=90,
    )
    if r == 0 and out:
        return True, f"vllm {out.splitlines()[-1]} + lm_eval 已就绪"
    return False, "vllm/lm_eval 未安装（点「一键装评测环境」）"

def fetch_eval_results(outdir: str) -> list[dict]:
    """抓本机 lm_eval 结果 json，顺序固定：[量化, 基线]。
    结果落在 task_dir/eval_*/quant/ 与 eval_*/baseline/ 子目录；
    各取最新一次评测（eval_* 目录按名称时间戳倒序）里最新的结果文件。"""
    od = q_remote(outdir)

    def latest(pattern_glob: str) -> str | None:
        r, out, _ = run_shell(f"ls -1t {od}/{pattern_glob} 2>/dev/null | head -1", timeout=20)
        f = out.strip()
        return f if r == 0 and f else None

    # 最新一次评测的 eval_* 目录（目录名含时间戳，按名称倒序 = 最新在前）
    r, out, _ = run_shell(
        f"ls -1 {od}/eval_* 2>/dev/null | sort -r | head -1", timeout=20)
    latest_eval = out.strip() if r == 0 and out.strip() else None
    if latest_eval:
        qf = latest(f"{latest_eval}/quant/results_*.json")
        bf = latest(f"{latest_eval}/baseline/results_*.json")
    else:  # 兼容旧布局：结果直接落在 task_dir 下
        qf = latest("results_*.json")
        bf = None
    files: list[dict] = []
    for f in (qf, bf):
        if not f:
            continue
        r2, j, _ = run_shell(f"cat {q_remote(f)}", timeout=60)
        if r2 == 0 and j:
            try:
                import json as _json
                files.append({"file": f.rsplit('/', 1)[-1], "data": _json.loads(j)})
            except Exception:
                _log.warning("results 文件解析失败，已跳过: %s", f, exc_info=True)
                continue
    return files

# ---------------------------------------------------------------- 任务

@dataclass
class Task:
    task_id: str
    remote_cmd: str
    log_path: str
    pid: str = ""
    state: str = "starting"  # starting | running | done | failed | aborted
    started_at: float = field(default_factory=time.time)
    death_grace: int = 0  # 进程已死但成功 marker 未落盘时的宽限轮数（>0 = 宽限中，不判 failed）
    done_marker: str = "Quantized model exported to"  # 日志里出现 = 成功结束
    pkill_pattern: str = ""  # 中止时顺带杀掉的子进程（如 vllm serve）

def start_task(cfg: AppConfig, task_dir: str, model: str, export_dir: str,
               remote_cmd: str, proxy_env: str, log) -> Task:
    ts = time.strftime("%Y%m%d-%H%M%S")
    task_dir_full = f"{task_dir.rstrip('/')}/{model.replace('/', '__')}_{ts}"
    log_path = f"{task_dir_full}/run.log"
    cmd = (
        f"mkdir -p {q_remote(task_dir_full)} && cd {q_remote(cfg.local.repo)} && "
        f"export PATH={q_remote(cfg.local.venv)}/bin:$PATH && {proxy_env} "
        f"nohup bash -c {shlex.quote(remote_cmd)} > {q_remote(log_path)} 2>&1 & echo $!"
    )
    r, out, err = run_shell(cmd, timeout=30, env_extra=hf_task_env(cfg))
    if r != 0:
        log(f"任务启动失败: {err}")
        raise RuntimeError(err)
    pid = out.splitlines()[-1].strip()
    log(f"任务已提交 pid={pid} 日志: {log_path}")
    return Task(task_id=f"{model}_{ts}", remote_cmd=remote_cmd,
                log_path=log_path, pid=pid, state="running")

def start_script_task(cfg: AppConfig, task_dir_full: str, script: str,
                      proxy_env: str, log, pkill_pattern: str = "") -> Task:
    """把 shell 脚本写到本机任务目录再 nohup 执行（评测流水线用）。
    与 start_task 的区别：命令先落盘成 run.sh，避免多层 bash -c 引号嵌套。"""
    log_path = f"{task_dir_full}/run.log"
    script_path = f"{task_dir_full}/run.sh"
    write_cmd = (
        f"mkdir -p {q_remote(task_dir_full)} && "
        f"cat > {q_remote(script_path)} <<'__MO_WEBUI_EOF__'\n"
        f"{script}\n__MO_WEBUI_EOF__"
    )
    r, _, err = run_shell(write_cmd, timeout=30)
    if r != 0:
        log(f"脚本写入失败: {err}")
        raise RuntimeError(err)
    launch = (
        f"cd {q_remote(task_dir_full)} && "
        f"export PATH={q_remote(cfg.local.venv)}/bin:$PATH && {proxy_env}"
        f"nohup bash run.sh > run.log 2>&1 & echo $!"
    )
    r, out, err = run_shell(launch, timeout=30, env_extra=hf_task_env(cfg))
    if r != 0:
        log(f"任务启动失败: {err}")
        raise RuntimeError(err)
    pid = out.splitlines()[-1].strip()
    log(f"任务已提交 pid={pid} 脚本: {script_path}")
    return Task(task_id=task_dir_full.rsplit('/', 1)[-1], remote_cmd="bash run.sh",
                log_path=log_path, pid=pid, state="running",
                done_marker="EVAL_DONE", pkill_pattern=pkill_pattern)

# 进程已死、成功 marker 未出现时，最多再宽限多少轮才判 failed（轮询 5s/轮）
_DEATH_GRACE_ROUNDS = 6


def poll_task(task: Task) -> None:
    """探活并更新 task.state。

    两个必须避开的误判（2026-10-15 远程误判 failed 事故）：
    1. run_shell 返回 rc=-1 表示探测命令本身超时（系统繁忙），不是进程死了
       —— 旧代码 `if r != 0` 把超时当已死，进程还在跑就提前判 failed。
    2. 进程刚死时 stdout 块缓冲的最后一行可能还没落盘（或包装 shell 先死），
       tail 里还没有成功 marker —— 立即判 failed 会把「正在收尾的成功任务」
       误判成失败。给 _DEATH_GRACE_ROUNDS 轮宽限，marker 落盘即翻 done。
    """
    r, out, _ = run_shell(f"kill -0 {shlex.quote(task.pid)} 2>/dev/null && echo alive", timeout=15)
    if r == -1:
        return  # 探测超时：状态未知，保持原状，下轮再探
    if r == 0:
        task.state = "running"
        task.death_grace = 0  # 还活着，宽限计数清零
        return
    # 进程确实死了：读日志尾部找成功 marker
    r2, tail, _ = run_shell(f"tail -c 4000 {q_remote(task.log_path)}", timeout=15)
    if r2 == -1:
        return  # 读日志超时：下轮再读（空串不能当「日志为空」）
    if task.done_marker in tail:
        task.state = "done"
        return
    if not tail.strip():
        task.state = "done"  # 日志为空且进程已死：沿用旧行为（无输出 = 无实质失败信息）
        return
    # 死了但 marker 还没出现：进入/继续宽限
    if task.death_grace <= 0:
        task.death_grace = _DEATH_GRACE_ROUNDS
        return
    task.death_grace -= 1
    if task.death_grace == 0:
        task.state = "failed"

def read_log(task: Task, tail_bytes: int = 20000) -> str:
    return tail_log(task.log_path, tail_bytes)

def tail_log(path: str, tail_bytes: int = 20000) -> str:
    """读本机日志尾部（进度解析用）。读不到就返空串，调用方当「暂无输出」。"""
    r, out, _ = run_shell(f"tail -c {int(tail_bytes)} {q_remote(path)} 2>/dev/null",
                          timeout=15)
    return out or ""

def abort_task(task: Task) -> str:
    r, _, _ = run_shell(f"kill -TERM {shlex.quote(task.pid)} 2>/dev/null; echo killed", timeout=15)
    if task.pkill_pattern:
        # 顺带杀掉脚本拉起的 vllm serve 等子进程，避免占着显存
        run_shell(f"pkill -f {shlex.quote(task.pkill_pattern)} 2>/dev/null; echo ok", timeout=15)
    task.state = "aborted"
    return "已发送中止信号（已连带清理 vllm serve）" if task.pkill_pattern else "已发送中止信号"

def export_file_listing(task: Task, log_path_hint: str = "") -> str:
    """列出本机导出目录内容（任务完成后用于结果校验区）。"""
    m = re.search(r"Quantized model exported to:\s*(\S+)", read_log(task, 5000))
    target = (m.group(1) if m else "").rstrip(".")
    if not target:
        target = task.log_path.rsplit("/", 1)[0]  # 回退到任务目录
    r, out, _ = run_shell(f"ls -lh {q_remote(target)} 2>/dev/null | head -40", timeout=20)
    return out or f"未找到导出目录 {target}（可从日志手动确认）"

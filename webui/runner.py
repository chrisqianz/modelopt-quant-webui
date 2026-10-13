"""远程 SSH 执行器：状态探测 / venv / 代理隧道 / 任务生命周期。

所有远程操作都是只读探测或用户显式发起的任务；GPU 占用不抢占、不改动。
"""

from __future__ import annotations

import getpass
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import progress
from .config import HostConfig, NetConfig, PROXY_AUTO, PROXY_DIRECT, PROXY_MANUAL

SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    # 新主机首次自动记录指纹：密码认证时 sshpass 只认密码提示，卡在指纹确认会挂死
    "-o", "StrictHostKeyChecking=accept-new",
]

# BatchMode 只对密钥主机加：它会禁掉密码提示，密码主机加了就等于永远登不进去。
SSHPASS_HINT = ("没装 sshpass，密码登录用不了。macOS 装一下："
                "brew install hudochenkov/sshpass/sshpass")


def q_remote(path: str, host: HostConfig | None = None) -> str:
    """远程路径引用。shlex.quote 会把 ~ 括进单引号导致不展开，
    所以前导 ~ 转成绝对路径（host 已知时解析 $HOME，否则用 ${HOME}，
    后者仅限不会被二次引用的探测命令）。"""
    p = path.strip()
    home = "${HOME}"
    if p.startswith("~/") and host is not None:
        home = remote_home(host)
    elif p == "~" and host is not None:
        return remote_home(host)
    if not p.startswith("~"):
        return shlex.quote(p)
    rest = p[2:] if p.startswith("~/") else ""
    return f"{home}" if not rest else f"{home}/{shlex.quote(rest)}"


def remote_home(host: HostConfig) -> str:
    """解析执行侧的 $HOME（缓存，一次往返）。"""
    if host.is_local:
        return os.path.expanduser("~")
    key = f"{host.ssh}:{host.ssh_port}"
    if key not in _REMOTE_HOME:
        r, out, _ = run_remote(host, "echo $HOME", timeout=10)
        _REMOTE_HOME[key] = out if r == 0 and out.startswith("/") else "~"
    return _REMOTE_HOME[key]


_REMOTE_HOME: dict[str, str] = {}

_TUNNELS: dict[str, tuple[int, str]] = {}   # host name -> (远程侧端口, 远程侧代理 scheme)
_HF_DIRECT: dict[str, tuple[bool, float]] = {}   # host name -> (直连可达, 缓存过期时刻)


def _host_key(host: HostConfig) -> str:
    return f"{host.ssh}:{host.ssh_port}"


def _ssh_error(err: str) -> str:
    """把 ssh 的英文报错翻成能行动的一句话。"""
    e = (err or "").strip()
    low = e.lower()
    if "permission denied" in low or "authentication failed" in low:
        return "认证失败：用户名或密码不对"
    if "connection refused" in low or "no route to host" in low or "host down" in low:
        return "连不上：端口没开 / 机器没开机 / IP 不对"
    if "timed out" in low or "timeout" in low:
        return "连接超时：IP 不可达或被防火墙拦"
    if "could not resolve" in low:
        return "域名解析失败：ssh 地址写错了？"
    if "host key" in low or "known hosts" in low or "man in the middle" in low:
        return "主机指纹变了（重装系统/换机器）：本机执行 ssh-keygen -R <host> 后重试"
    return e[:300] or "SSH 连接失败"


def _is_conn_err(err: str) -> bool:
    low = (err or "").lower()
    return any(k in low for k in (
        "broken pipe", "connection closed", "connection reset", "connection refused",
        "timed out", "mux_client_request_session", "ssh_exchange_identification",
        "control socket", "port forwarding failed", "closed by remote host",
        "could not connect", "no such file or directory"))


# ---------------------------------------------------------------- SSH 连接复用

_MASTER_OK: dict[str, float] = {}   # host key -> 「确认存活」缓存到几点（monotonic）
_CTL_NAME_LEN = 11                  # 形如 a1b2c3d4.sk
_CTL_DIR: Path | None = None


def _ctl_dir() -> Path:
    """mux socket 目录，优先 /tmp。

    macOS 的 TMPDIR 是 /var/folders/50/…/T/，光目录就 60 字节，加哈希名再让 ssh
    绑定时自己追加的随机后缀（.atV5YoLRCgxK16Hw）一顶，就超过 Unix domain socket
    的 104 字节上限，ControlMaster 直接起不来：too long for Unix domain socket。
    """
    global _CTL_DIR
    if _CTL_DIR is not None:
        return _CTL_DIR
    for base in (Path("/tmp"), Path(tempfile.gettempdir())):
        cand = base / f"moq-{os.getuid()}"
        try:
            cand.mkdir(parents=True, exist_ok=True)
            os.chmod(cand, 0o700)
        except OSError:
            continue
        # 104 是 sun_path 上限，24 是留给 ssh 追加的随机后缀 + 换机余量
        if len(str(cand)) + _CTL_NAME_LEN + 24 < 104:
            _CTL_DIR = cand
            return cand
    raise RuntimeError("找不到可写的短路径目录放 SSH 复用 socket（/tmp 不可写？）")


def control_path(host: HostConfig) -> str:
    """每个 user@host:port 一条 mux socket。名字压到 11 字节以避开 104 上限。"""
    digest = hashlib.md5(_host_key(host).encode()).hexdigest()[:8]
    return str(_ctl_dir() / f"{digest}.sk")


def sshpass_bin() -> str:
    found = shutil.which("sshpass")
    if found:
        return found
    for cand in ("/usr/local/bin/sshpass", "/opt/homebrew/bin/sshpass"):
        if Path(cand).exists():
            return cand
    return ""


def _port_args(host: HostConfig) -> list[str]:
    return ["-p", str(host.ssh_port)] if host.ssh_port != 22 else []


def _ssh_base(host: HostConfig, *, mux: bool = True) -> list[str]:
    """ssh 命令前缀（不含远程命令）。mux=True 时挂到 ControlMaster 长连接上。"""
    opts = list(SSH_OPTS)
    if not host.password:
        opts += ["-o", "BatchMode=yes"]
    if mux:
        opts += ["-o", f"ControlPath={control_path(host)}",
                 "-o", "ControlMaster=auto", "-o", "ControlPersist=600"]
    return ["ssh", *opts, *_port_args(host), host.ssh]


def master_alive(host: HostConfig) -> bool:
    if host.is_local:
        return True
    r, _, _ = run_local(["ssh", "-o", f"ControlPath={control_path(host)}", "-O", "check",
                         *_port_args(host), host.ssh], timeout=10)
    return r == 0


def ensure_master(host: HostConfig, *, force: bool = False) -> str:
    """拉起一条 ControlPersist 长连接，之后所有 ssh 复用，只认证一次。
    密码主机必须先有 master：本程序没有 tty，ssh 自己没法问密码，只能 sshpass 喂给 master。
    返回 "" = 可用，否则是给用户看的错误说明。"""
    if host.is_local:
        return ""
    key = _host_key(host)
    if not force and _MASTER_OK.get(key, 0) > time.monotonic():
        return ""
    if not force and master_alive(host):
        _MASTER_OK[key] = time.monotonic() + 30
        return ""

    ctl = ["-o", f"ControlPath={control_path(host)}", "-o", "ControlMaster=yes",
           "-o", "ControlPersist=3600"]
    argv = ["ssh", *ctl, *SSH_OPTS, *_port_args(host)]
    if host.password:
        argv += ["-o", "NumberOfPasswordPrompts=1"]
    argv += ["-N", "-f", host.ssh]
    if not host.password:
        # 密钥登录：建不起来也不影响后续命令（ssh 自动退回直连），静默试一次
        run_local(argv, timeout=20)
        return ""

    sp = sshpass_bin()
    if not sp:
        return SSHPASS_HINT
    # 密码走 env，不进 argv：argv 会被 ps 看到
    r, _, err = run_local([sp, *argv], timeout=45, env={**os.environ, "SSHPASS": host.password})
    if r != 0:
        _MASTER_OK.pop(key, None)
        return _ssh_error(err)
    _MASTER_OK[key] = time.monotonic() + 30
    return ""


def drop_master(host: HostConfig) -> None:
    """主机配置改了（地址/端口/密码）后，断开旧连接并清掉相关缓存。"""
    key = _host_key(host)
    if not host.is_local:
        run_local(["ssh", "-o", f"ControlPath={control_path(host)}", "-O", "exit",
                   *_port_args(host), host.ssh], timeout=10)
    _MASTER_OK.pop(key, None)
    _REMOTE_HOME.pop(key, None)
    _TUNNELS.pop(host.name, None)
    _HF_DIRECT.pop(host.name, None)


def check_connection(host: HostConfig) -> tuple[bool, str]:
    """「测试连接」：验能不能进到目标执行环境，给一句人话结论。"""
    drop_master(host)              # 可能刚改过密码/地址，强制重新认证
    if host.is_local:
        rc, out, err = _exec(["bash", "-c", "echo __OK__; hostname"], timeout=15)
        if rc != 0 or "__OK__" not in out:
            return False, f"✗ 本机 shell 执行失败：{(err or out)[:200]}"
        return True, (f"✓ 本机执行可用（{os.name} / {getpass.getuser()}@{socket.gethostname()}"
                      f" · HOME={os.path.expanduser('~')}）"
                      + (f" · 主机名 {out.replace('__OK__', '').strip()}" if host.name else ""))
    if host.password:
        err = ensure_master(host, force=True)
        if err:
            return False, f"✗ {err}"
    t0 = time.monotonic()
    r, out, err = run_remote(host, "echo __OK__; hostname", timeout=25)
    if r != 0 or "__OK__" not in out:
        return False, f"✗ {_ssh_error(err)}"
    ms = int((time.monotonic() - t0) * 1000)
    name = out.replace("__OK__", "").strip()
    return True, (f"✓ {host.auth_kind}登录成功，{ms} ms（已建长连接复用）"
                  + (f" · 远端主机名 {name}" if name else ""))


# ---------------------------------------------------------------- 单次执行

def _exec(argv: list[str], timeout: int = 60,
          env: dict | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"命令超时 ({timeout}s): {' '.join(argv[-2:])[:120]}"


def run_remote(host: HostConfig, remote_cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """在目标机器上执行命令，返回 (returncode, stdout, stderr)。
    mode=local 直接起本地 shell（UI 部署在有 GPU 的机器上就是这条路）；
    mode=remote 走 ssh（有 master 就复用）。
    调用方自行负责后台化（nohup ... & echo $!），本函数只负责单次执行。
    长连接刚掉时重试一次（只认连接类报错，避免把远程命令重跑一遍）。"""
    if host.is_local:
        return _exec(["bash", "-c", remote_cmd], timeout=timeout)
    if host.password:
        err = ensure_master(host)
        if err:
            return -1, "", err
    rc, out, err = _exec([*_ssh_base(host), remote_cmd], timeout=timeout)
    if rc == 255 and err and _is_conn_err(err) and not master_alive(host):
        _MASTER_OK.pop(_host_key(host), None)
        err2 = ensure_master(host, force=True)
        if err2:
            return -1, "", err2
        rc, out, err = _exec([*_ssh_base(host), remote_cmd], timeout=timeout)
    return rc, out, err


def run_local(argv: list[str], timeout: int = 60,
              env: dict | None = None) -> tuple[int, str, str]:
    return _exec(argv, timeout=timeout, env=env)


# ---------------------------------------------------------------- 状态探测

@dataclass
class HostStatus:
    reachable: bool = False
    auth: str = ""
    gpu: str = ""
    gpu_util: int | None = None
    gpu_mem: str = ""
    gpu_busy: bool = False
    repo_ok: bool = False
    venv_ok: bool = False
    uv_ok: bool = False
    disk_free: str = ""
    hf_cache: str = ""
    hf_direct_ok: bool = False      # 远程直连 huggingface.co 是否可达
    hf_proxy_ok: bool = False       # 按当前代理配置能否访问 HF
    proxy_note: str = ""            # 实际会用哪条路下载 HF
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "可达": "✓" if self.reachable else "✗",
            "认证": self.auth or "—",
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
        return "✓ 远程直连可达"
    if st.hf_proxy_ok:
        return f"✓ 直连不通，走代理可达（{st.proxy_note}）"
    return f"⚠ 直连不通，代理未就绪（{st.proxy_note or '未配置'}）"


def probe_hf(host: HostConfig, proxy: str = "") -> tuple[bool, str]:
    """探测远程能否访问 huggingface.co。proxy 传 http:// 或 socks5h:// 地址，空 = 直连。
    返回 (可达, 说明)。"""
    env = _proxy_env(proxy) if proxy else ""
    cmd = (f"{env}curl -s -o /dev/null -m 15 -w '%{{http_code}} %{{time_total}}' "
           f"https://huggingface.co/api/models")
    r, out, err = run_remote(host, cmd, timeout=30)
    if r != 0:
        return False, _ssh_error(err)
    code, _, secs = out.partition(" ")
    detail = f"HTTP {code}"
    try:
        detail += f"，{float(secs):.1f}s"
    except ValueError:
        pass
    if code != "200" and err:
        detail += f"（{err.splitlines()[-1][:80]}）"
    return code == "200", detail


def hf_direct_ok(host: HostConfig, ttl: float = 300.0) -> bool:
    """远程直连可达性（探一次缓存 ttl 秒；每次刷状态都真探一遍太慢）。"""
    cached = _HF_DIRECT.get(host.name)
    if cached and cached[1] > time.monotonic():
        return cached[0]
    ok, _ = probe_hf(host)
    _HF_DIRECT[host.name] = (ok, time.monotonic() + ttl)
    return ok


def probe_host(host: HostConfig, app_cfg=None) -> HostStatus:
    st = HostStatus()
    r, out, err = run_remote(host, "echo __OK__", timeout=15)
    if r != 0 or "__OK__" not in out:
        st.error = err or "SSH 连接失败"
        return st
    st.reachable = True

    # GPU（只读 nvidia-smi，不产生任何负载）
    r, out, _ = run_remote(host, r"nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1", timeout=15)
    if r == 0 and out:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            st.gpu, st.gpu_mem = parts[0], f"{re.sub(r'[^0-9]',' ',parts[2]).strip()}/{re.sub(r'[^0-9]',' ',parts[3]).strip()} MiB"
            st.gpu_util = int(re.sub(r"[^0-9]", "", parts[1]) or 0)
            st.gpu_busy = st.gpu_util >= 80

    # 环境
    r, out, _ = run_remote(host, "command -v uv || ls ~/.local/bin/uv 2>/dev/null", timeout=10)
    st.uv_ok = r == 0 and bool(out)

    r, out, _ = run_remote(host, f"test -d {q_remote(host.remote_repo, host)}/examples/hf_ptq && echo ok", timeout=10)
    st.repo_ok = out == "ok"

    r, out, _ = run_remote(host, f"test -x {q_remote(host.venv_dir, host)}/bin/python && echo ok", timeout=10)
    st.venv_ok = out == "ok"

    r, out, _ = run_remote(host, "df -h ~ | awk 'NR==2{print $4}'", timeout=10)
    st.disk_free = out

    remote_home(host)  # 预热缓存，任务命令拼绝对路径用

    r, out, _ = run_remote(host, f"du -sh {q_remote(host.hf_cache, host)} 2>/dev/null | awk '{{print $1}}'", timeout=20)
    st.hf_cache = out

    # HF 可达性：国内直连常不通，按网络面板的模式逐条验证
    if app_cfg is not None:
        st.auth = host.auth_kind
        st.hf_direct_ok = hf_direct_ok(host)
        net: NetConfig = app_cfg.net
        if st.hf_direct_ok:
            st.proxy_note = "直连"
        elif net.mode == PROXY_DIRECT:
            st.proxy_note = "已设为「不使用代理」"
        elif net.mode == PROXY_MANUAL:
            url = normalize_proxy(net.remote_proxy)
            if url:
                st.hf_proxy_ok, st.proxy_note = probe_hf(host, proxy=url)
            else:
                st.proxy_note = "未填「远程代理地址」"
        elif net.mode == PROXY_AUTO:
            port = ensure_hf_tunnel(app_cfg, host)
            if port:
                url = f"{_TUNNELS[host.name][1]}://127.0.0.1:{port}"
                st.hf_proxy_ok, detail = probe_hf(host, proxy=url)
                st.proxy_note = f"隧道 → 本机代理 {detail}"
            else:
                st.proxy_note = "隧道未起来，检查「本机代理地址」"
    return st


# ---------------------------------------------------------------- 代理隧道

def normalize_proxy(url: str) -> str:
    """把用户手填的代理地址成能直接注进 env 的形式：
    socks5:// → socks5h://（让代理端解 DNS，远程本机常解不了 huggingface.co）；
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


def tunnel_scheme(host: HostConfig) -> str:
    """隧道远程侧该用什么 scheme —— 取决于本机那个端口说的是哪种协议。"""
    got = _TUNNELS.get(host.name)
    return got[1] if got else "http"


def _remote_port_open(host: HostConfig, port: int) -> bool:
    """远程侧 127.0.0.1:port 是否真在听（-R 成败的唯一可信判据）。"""
    r, out, _ = run_remote(host,
                           f"bash -c 'exec 3<>/dev/tcp/127.0.0.1/{port} && echo open' 2>/dev/null",
                           timeout=15)
    return r == 0 and "open" in out


def ensure_hf_tunnel(app_cfg, host: HostConfig) -> int:
    """建/复用 remote -> local 反向隧道，返回远程侧端口（0 = 起不来）。
    只有「SSH 远程机器 + 自动模式 + 直连不通 + 填了本机代理」才会走到这里；
    本机执行不需要隧道——本机的 127.0.0.1 代理本来就在同一台机器上。
    经典 -R 冒号语法（本机 ssh 对 -o RemoteForward 解析异常，实测不可用）；
    挂在 ControlMaster 上时只加转发、不再走一遍认证。"""
    if host.is_local:
        return 0
    net: NetConfig = getattr(app_cfg, "net", None) or NetConfig()
    parsed = _parse_proxy_addr(net.local_proxy)
    if not parsed:
        return 0
    scheme, lhost, lport = parsed
    # 稳定端口：md5 前 3 字节映射到 10800-11999，避免随机 hash 跨进程漂移
    port = int(hashlib.md5(host.name.encode()).hexdigest()[:3], 16) % 1200 + 10800
    remote_scheme = "socks5h" if scheme.startswith("socks") else "http"
    if _TUNNELS.get(host.name, (0, ""))[0] == port and master_alive(host):
        return port                      # 长连接还活着，转发就还在
    if host.password and ensure_master(host):
        return 0
    argv = [*_ssh_base(host), "-N", "-f", "-q", "-R", f"{port}:{lhost}:{lport}"]
    run_local(argv, timeout=30)   # 同一 master 上重复加同端口会报错，以实连结果为准
    if not _remote_port_open(host, port):
        return 0
    _TUNNELS[host.name] = (port, remote_scheme)
    return port


def proxy_url_for(app_cfg, host: HostConfig) -> tuple[str, str]:
    # 返回 (要注入的 proxy URL, 给人看的说明)；URL 为空表示不注入"
    net: NetConfig = getattr(app_cfg, "net", None) or NetConfig()
    mine = net.local_proxy if host.is_local else net.remote_proxy
    where = "本机" if host.is_local else "远程"
    if net.mode == PROXY_DIRECT:
        return "", "按设置不用代理"
    if net.mode == PROXY_MANUAL:
        url = normalize_proxy(mine)
        if not url:
            return "", f"强制走代理但{where}代理地址没填，暂时不注入"
        return url, f"按设置走{where}代理 {url}"
    # auto：直连优先，不通再找代理
    if hf_direct_ok(host):
        return "", "直连可达，不注入代理"
    url = normalize_proxy(mine)
    if url:
        return url, f"直连不通，走{where}代理 {url}"
    if not host.is_local:
        port = ensure_hf_tunnel(app_cfg, host)
        if port:
            url = f"{tunnel_scheme(host)}://127.0.0.1:{port}"
            return url, f"直连不通，SSH 隧道接本机代理（远程 127.0.0.1:{port}）"
    return "", "直连不通，而且没填可用代理，下载大概率会失败"


def proxy_for(app_cfg, host: HostConfig) -> str:
    """按网络面板的配置，返回要注入执行环境（nohup 命令前缀）的 proxy 变量串。"""
    url, _ = proxy_url_for(app_cfg, host)
    return _proxy_env(url) if url else ""


def test_proxy(app_cfg, host: HostConfig, url: str = "") -> tuple[bool, str]:
    """「测试网络」：按当前配置（或 url 指定的地址）从执行侧实打一次 huggingface.co。"""
    net: NetConfig = getattr(app_cfg, "net", None) or NetConfig()
    url = normalize_proxy(url)
    if url:
        return probe_hf(host, proxy=url)
    if net.mode == PROXY_DIRECT:
        ok, detail = probe_hf(host)
        return ok, ("直连可达" if ok else f"直连不通：{detail}")
    if net.mode == PROXY_MANUAL:
        url = normalize_proxy(net.local_proxy if host.is_local else net.remote_proxy)
        if not url:
            where = "本机" if host.is_local else "远程"
            return False, f"{where}代理地址是空的，先填一个这台机器连得到的地址"
        ok, detail = probe_hf(host, proxy=url)
        return ok, (f"{url} 可用（{detail}）" if ok else f"{url} 不通：{detail}")
    if hf_direct_ok(host):
        return True, "直连 huggingface.co 可达，不需要代理"
    use, note = proxy_url_for(app_cfg, host)
    if not use:
        return False, f"直连不通且无代理可用：{note}"
    ok, detail = probe_hf(host, proxy=use)
    return ok, (f"{note} · 实打成功（{detail}）" if ok
                else f"{note} · 但 HF 取不到：{detail}")


def _proxy_env(proxy: str) -> str:
    p = shlex.quote(normalize_proxy(proxy))
    return (f"export http_proxy={p} https_proxy={p} "
            f"HF_HUB_DISABLE_IMPLICIT_TOKEN=1 no_proxy=${{no_proxy:+$no_proxy,}}localhost,127.0.0.1;")


# ---------------------------------------------------------------- venv

def _resolve_uv(host: HostConfig) -> str:
    r, out, _ = run_remote(host, "command -v uv || echo ~/.local/bin/uv", timeout=10)
    return out.splitlines()[-1] if r == 0 and out else "~/.local/bin/uv"


def ensure_venv(host: HostConfig, log, app_cfg=None) -> None:
    """在远程创建 venv 并安装 modelopt[hf]。长耗时操作走 nohup，轮询日志。
    app_cfg 提供时顺带注入 HF 代理隧道（国内 pypi/HF 都可能受限，统一走代理更稳）。"""
    venv = q_remote(host.venv_dir, host)
    r, out, _ = run_remote(host, f"test -x {venv}/bin/python && echo ok", timeout=10)
    if out == "ok":
        log("venv 已存在，跳过创建")
        return
    uv = _resolve_uv(host)
    repo = q_remote(host.remote_repo, host)
    proxy_env = proxy_for(app_cfg, host) if app_cfg is not None else ""
    cmd = (
        f"mkdir -p {venv} && "
        f"{proxy_env}"
        f"{uv} venv {venv} --python 3.12 && "
        f"cd {repo} && "
        f"VIRTUAL_ENV={venv} {uv} pip install -e '.[hf]' 2>&1; "
        f"echo VENV_DONE rc=$?"
    )
    log_path = f"/tmp/modelopt_venv_setup_{int(time.time())}.log"
    bg_cmd = f"nohup bash -c {shlex.quote(cmd)} > {shlex.quote(log_path)} 2>&1 & echo $!"
    r, out, err = run_remote(host, bg_cmd, timeout=30)
    if r != 0:
        log(f"venv 创建启动失败: {err}")
        return
    log("venv 创建中（uv venv + pip install -e '.[hf]'，通常 5-15 分钟；"
        + ("已走代理" if proxy_env else "直连安装") + "）…")
    _wait_for_marker(host, log_path, "VENV_DONE", log, wait_s=15, msg="…仍在安装依赖",
                     parser=progress.parse_install)


def ensure_eval_env(host: HostConfig, log, app_cfg=None) -> None:
    """在已有 venv 里加装评测栈：vllm + lm_eval。长耗时操作走 nohup，轮询日志。"""
    venv = q_remote(host.venv_dir, host)
    r, out, _ = run_remote(host, f"test -x {venv}/bin/python && echo ok", timeout=10)
    if out != "ok":
        log("⚠ 基础 venv 不存在，请先点状态栏的「一键创建 venv」")
        return
    uv = _resolve_uv(host)
    proxy_env = proxy_for(app_cfg, host) if app_cfg is not None else ""
    cmd = (
        f"{proxy_env}"
        f"VIRTUAL_ENV={venv} {uv} pip install 'vllm>=0.10.1' 'lm_eval[ifeval]>=0.4.12,<0.5' 2>&1; "
        f"echo EVAL_ENV_DONE rc=$?"
    )
    log_path = f"/tmp/modelopt_eval_env_{int(time.time())}.log"
    bg_cmd = f"nohup bash -c {shlex.quote(cmd)} > {shlex.quote(log_path)} 2>&1 & echo $!"
    r, out, err = run_remote(host, bg_cmd, timeout=30)
    if r != 0:
        log(f"评测环境安装启动失败: {err}")
        return
    log("评测环境安装中（vllm + lm_eval，下载体积大，通常 10-30 分钟；"
        + ("已走代理" if proxy_env else "直连安装") + "）…")
    _wait_for_marker(host, log_path, "EVAL_ENV_DONE", log, wait_s=20, msg="…仍在安装依赖",
                     parser=progress.parse_install)


def _wait_for_marker(host: HostConfig, log_path: str, marker: str, log, wait_s: int,
                     msg: str, parser=None) -> None:
    while True:
        r, out, _ = run_remote(host, f"tail -c 8000 {shlex.quote(log_path)} 2>/dev/null",
                               timeout=15)
        if marker in (out or ""):
            log("完成" if "rc=0" in out else f"失败（见 {log_path}）: {out}")
            return
        time.sleep(wait_s)
        rep = parser((out or "").splitlines()) if parser else None
        log(rep.text if rep and rep.text else msg)


# ---------------------------------------------------------------- 评测

def probe_eval_env(host: HostConfig) -> tuple[bool, str]:
    """探测远程 venv 里是否已装 vllm + lm_eval（只读，不占 GPU）。"""
    venv = q_remote(host.venv_dir, host)
    r, out, _ = run_remote(host, f"test -x {venv}/bin/python && echo ok", timeout=10)
    if out != "ok":
        return False, "基础 venv 未创建（先点状态栏「一键创建 venv」）"
    r, out, _ = run_remote(
        host,
        f"{venv}/bin/python -c 'import vllm, lm_eval; print(vllm.__version__)' 2>/dev/null",
        timeout=90,
    )
    if r == 0 and out:
        return True, f"vllm {out.splitlines()[-1]} + lm_eval 已就绪"
    return False, "vllm/lm_eval 未安装（点「一键装评测环境」）"


def fetch_eval_results(host: HostConfig, outdir: str) -> list[dict]:
    """抓远程 lm_eval 结果 json，顺序固定：[量化, 基线]。
    结果落在 task_dir/eval_*/quant/ 与 eval_*/baseline/ 子目录；
    各取最新一次评测（eval_* 目录按名称时间戳倒序）里最新的结果文件。"""
    od = q_remote(outdir, host)

    def latest(pattern_glob: str) -> str | None:
        r, out, _ = run_remote(host, f"ls -1t {od}/{pattern_glob} 2>/dev/null | head -1", timeout=20)
        f = out.strip()
        return f if r == 0 and f else None

    # 最新一次评测的 eval_* 目录（目录名含时间戳，按名称倒序 = 最新在前）
    r, out, _ = run_remote(
        host, f"ls -1 {od}/eval_* 2>/dev/null | sort -r | head -1", timeout=20)
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
        r2, j, _ = run_remote(host, f"cat {q_remote(f, host)}", timeout=60)
        if r2 == 0 and j:
            try:
                import json as _json
                files.append({"file": f.rsplit('/', 1)[-1], "data": _json.loads(j)})
            except Exception:
                continue
    return files


# ---------------------------------------------------------------- 任务

@dataclass
class Task:
    task_id: str
    host: HostConfig
    remote_cmd: str
    log_path: str
    pid: str = ""
    state: str = "starting"  # starting | running | done | failed | aborted
    started_at: float = field(default_factory=time.time)
    done_marker: str = "Quantized model exported to"  # 日志里出现 = 成功结束
    pkill_pattern: str = ""  # 中止时顺带杀掉的子进程（如 vllm serve）


def start_task(host: HostConfig, task_dir: str, model: str, export_dir: str,
               remote_cmd: str, proxy_env: str, log) -> Task:
    ts = time.strftime("%Y%m%d-%H%M%S")
    task_dir_full = f"{host.task_dir.rstrip('/')}/{model.replace('/', '__')}_{ts}"
    log_path = f"{task_dir_full}/run.log"
    cmd = (
        f"mkdir -p {q_remote(task_dir_full, host)} && cd {q_remote(host.remote_repo, host)} && "
        f"export PATH={q_remote(host.venv_dir, host)}/bin:$PATH && {proxy_env} "
        f"nohup bash -c {shlex.quote(remote_cmd)} > {q_remote(log_path, host)} 2>&1 & echo $!"
    )
    r, out, err = run_remote(host, cmd, timeout=30)
    if r != 0:
        log(f"任务启动失败: {err}")
        raise RuntimeError(err)
    pid = out.splitlines()[-1].strip()
    log(f"任务已提交 pid={pid} 日志: {log_path}")
    return Task(task_id=f"{model}_{ts}", host=host, remote_cmd=remote_cmd,
                log_path=log_path, pid=pid, state="running")


def start_script_task(host: HostConfig, task_dir_full: str, script: str,
                      proxy_env: str, log, pkill_pattern: str = "") -> Task:
    """把 shell 脚本写到远程任务目录再 nohup 执行（评测流水线用）。
    与 start_task 的区别：命令先落盘成 run.sh，避免多层 bash -c 引号嵌套。"""
    log_path = f"{task_dir_full}/run.log"
    script_path = f"{task_dir_full}/run.sh"
    write_cmd = (
        f"mkdir -p {q_remote(task_dir_full, host)} && "
        f"cat > {q_remote(script_path, host)} <<'__MO_WEBUI_EOF__'\n"
        f"{script}\n__MO_WEBUI_EOF__"
    )
    r, _, err = run_remote(host, write_cmd, timeout=30)
    if r != 0:
        log(f"脚本写入失败: {err}")
        raise RuntimeError(err)
    launch = (
        f"cd {q_remote(task_dir_full, host)} && "
        f"export PATH={q_remote(host.venv_dir, host)}/bin:$PATH && {proxy_env}"
        f"nohup bash run.sh > run.log 2>&1 & echo $!"
    )
    r, out, err = run_remote(host, launch, timeout=30)
    if r != 0:
        log(f"任务启动失败: {err}")
        raise RuntimeError(err)
    pid = out.splitlines()[-1].strip()
    log(f"任务已提交 pid={pid} 脚本: {script_path}")
    return Task(task_id=task_dir_full.rsplit('/', 1)[-1], host=host, remote_cmd="bash run.sh",
                log_path=log_path, pid=pid, state="running",
                done_marker="EVAL_DONE", pkill_pattern=pkill_pattern)


def poll_task(task: Task) -> None:
    r, out, _ = run_remote(task.host, f"kill -0 {shlex.quote(task.pid)} 2>/dev/null && echo alive", timeout=15)
    if r != 0:
        r2, tail, _ = run_remote(task.host, f"tail -c 4000 {q_remote(task.log_path, task.host)}", timeout=15)
        task.state = "done" if task.done_marker in tail or not tail.strip() else "failed"
        return
    task.state = "running"


def read_log(task: Task, tail_bytes: int = 20000) -> str:
    return remote_tail(task.host, task.log_path, tail_bytes)


def remote_tail(host: HostConfig, path: str, tail_bytes: int = 20000) -> str:
    """读远程日志尾部（进度解析用）。读不到就返空串，调用方当「暂无输出」。"""
    r, out, _ = run_remote(host, f"tail -c {int(tail_bytes)} {q_remote(path, host)} 2>/dev/null",
                           timeout=15)
    return out or ""


def remote_dir_summary(host: HostConfig, path: str) -> str:
    """远程目录一行摘要（HF 缓存下载时看进进没进）。"""
    p = q_remote(path, host)
    script = (f"d={p}; if [ -d $d ]; then "
              f"sz=$(du -sh $d 2>/dev/null | cut -f1); "
              f"n=$(find $d -type f 2>/dev/null | wc -l); "
              f"echo $sz / $n; else echo MISSING; fi")
    r, out, _ = run_remote(host, script, timeout=90)
    if r != 0 or not out or "MISSING" in out:
        return "目录还不存在（首次下载会自动建）"
    size, _, count = out.partition("/")
    return f"{size.strip()} · {count.strip()} 个文件"


def abort_task(task: Task) -> str:
    r, _, _ = run_remote(task.host, f"kill -TERM {shlex.quote(task.pid)} 2>/dev/null; echo killed", timeout=15)
    if task.pkill_pattern:
        # 顺带杀掉脚本拉起的 vllm serve 等子进程，避免占着显存
        run_remote(task.host, f"pkill -f {shlex.quote(task.pkill_pattern)} 2>/dev/null; echo ok", timeout=15)
    task.state = "aborted"
    return "已发送中止信号（已连带清理 vllm serve）" if task.pkill_pattern else "已发送中止信号"


def export_file_listing(task: Task, log_path_hint: str = "") -> str:
    """列出远程导出目录内容（任务完成后用于结果校验区）。"""
    m = re.search(r"Quantized model exported to:\s*(\S+)", read_log(task, 5000))
    target = (m.group(1) if m else "").rstrip(".")
    if not target:
        target = task.log_path.rsplit("/", 1)[0]  # 回退到任务目录
    r, out, _ = run_remote(task.host, f"ls -lh {q_remote(target, task.host)} 2>/dev/null | head -40", timeout=20)
    return out or f"未找到导出目录 {target}（可从日志手动确认）"

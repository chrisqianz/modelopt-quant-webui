"""配置加载。

两层来源：
- config.yaml：server 监听 + defaults（可选，缺省即默认值）；network 段只作代理配置的初始种子。
- hosts.json：UI 维护的「主机列表 + 网络代理」，UI 保存后覆盖 yaml 种子。

hosts.json 结构：{"hosts": [...], "network": {...}}。
为兼容早期版本，文件若是一个裸 list，则视为仅主机列表。

注意：主机的 password 以明文存在 hosts.json（本机自用工具），落盘即 chmod 600。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent

# UI 维护的主机 + 网络配置持久化位置（与 config.yaml 同目录）
HOSTS_JSON = CONFIG_DIR / "hosts.json"

DEFAULT_REPO = "~/Model-Optimizer"
DEFAULT_VENV = "~/.venvs/modelopt"
DEFAULT_TASK_DIR = "~/quant-webui-tasks"
DEFAULT_HF_CACHE = "~/.cache/huggingface"

# 执行位置：本机 = UI 跑在有 GPU 的机器上（部署目标）；远程 = 开发机远程驱动
MODE_LOCAL = "local"
MODE_REMOTE = "remote"
EXEC_MODES = [("本机（UI 就跑在这台机器上）", MODE_LOCAL),
              ("SSH 远程机器", MODE_REMOTE)]

# 代理模式：三种互斥取值，UI 下拉直接对应。地址一律手填，不写死。
PROXY_AUTO = "auto"        # 先探直连，不通则回落到「本机代理地址」（远程主机则加 SSH 隧道）
PROXY_MANUAL = "manual"    # 直接用填的代理地址
PROXY_DIRECT = "direct"    # 不注入任何代理
PROXY_MODES = [
    ("自动（先试直连，不通再用代理）", PROXY_AUTO),
    ("强制走代理（用下面填的地址）", PROXY_MANUAL),
    ("不用代理（直连下载）", PROXY_DIRECT),
]


@dataclass
class HostConfig:
    name: str
    ssh: str = ""                      # mode=remote 时必填 user@host；本机留空
    mode: str = MODE_LOCAL
    ssh_port: int = 22
    remote_repo: str = DEFAULT_REPO
    venv_dir: str = DEFAULT_VENV
    task_dir: str = DEFAULT_TASK_DIR
    hf_cache: str = DEFAULT_HF_CACHE
    password: str = ""          # 留空 = 用 ssh 密钥/agent；非空 = sshpass 密码登录

    @property
    def is_local(self) -> bool:
        return self.mode == MODE_LOCAL

    @property
    def where(self) -> str:
        return "本机" if self.is_local else self.ssh

    @property
    def identity(self) -> str:
        return self.name if self.is_local else f"{self.name} ({self.ssh})"

    @property
    def auth_kind(self) -> str:
        if self.is_local:
            return "本机"
        return "密码" if self.password else "密钥"

    def to_dict(self) -> dict:
        return asdict(self)


# 不写死用户名/内网 IP：默认播种「本机」，移植到任何 GPU 机器开箱即用
_SEED = [HostConfig(name="本机（GPU 服务器）", ssh="", mode=MODE_LOCAL)]


@dataclass
class NetConfig:
    """HF 下载用的代理配置。mode 见 PROXY_MODES。"""
    mode: str = PROXY_AUTO
    remote_proxy: str = ""      # 远程主机用的代理地址（须是那台机器可达的，手填）
    local_proxy: str = ""       # 本机/跑 UI 这台机器的代理地址（手填，如 http://127.0.0.1:7890）

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AppConfig:
    server_host: str = "0.0.0.0"
    server_port: int = 7861
    hosts: list[HostConfig] = field(default_factory=list)
    net: NetConfig = field(default_factory=NetConfig)
    defaults: dict = field(default_factory=dict)

    def get_host(self, name: str) -> HostConfig:
        for h in self.hosts:
            if h.name == name:
                return h
        if not self.hosts:
            raise ValueError("还没有定义任何目标机器（「连接设置」面板添加）")
        if len(self.hosts) == 1:
            return self.hosts[0]
        raise ValueError(f"未知 host: {name}")

    def host_names(self) -> list[str]:
        return [h.name for h in self.hosts]


# ---------------------------------------------------------------- 持久化

def _read_state(path: Path) -> dict | None:
    """读 hosts.json，返回 {"hosts": [...], "network": {...}}；文件不存在返回 None。"""
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, list):          # 旧格式：裸主机列表
        return {"hosts": raw, "network": None}
    if isinstance(raw, dict):
        return {"hosts": raw.get("hosts") or [], "network": raw.get("network")}
    raise ValueError("hosts.json 格式无法识别")


def _write_state(hosts: list[HostConfig], net: NetConfig | None,
                 path: Path = HOSTS_JSON) -> None:
    payload = {"hosts": [h.to_dict() for h in hosts],
               "network": net.as_dict() if net else None}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    path.chmod(0o600)                  # 里面有明文密码


def load_hosts(path: Path = HOSTS_JSON) -> list[HostConfig]:
    """读主机列表；文件不存在则播种默认条目并落盘。"""
    try:
        state = _read_state(path)
    except Exception:
        return list(_SEED)
    if state is None:
        save_hosts(list(_SEED), path)
        return list(_SEED)
    hosts = []
    for h in state["hosts"]:
        try:
            # 旧条目没存过 mode：带 user@host 的一律仍当 SSH 远程，
            # 否则命令会在 UI 这台机器上跑（而不是到 GPU 机器），错得隐蔽
            if not str(h.get("mode") or "").strip():
                h = {**h, "mode": MODE_REMOTE if str(h.get("ssh") or "").strip() else MODE_LOCAL}
            hosts.append(HostConfig(**h))
        except TypeError:              # 老条目缺字段/多字段，逐个兜住
            continue
    return hosts or list(_SEED)


def load_net(path: Path = HOSTS_JSON, seed: NetConfig | None = None) -> NetConfig:
    """读网络代理配置；UI 没存过就用 seed（来自 config.yaml / env）。"""
    try:
        state = _read_state(path)
    except Exception:
        return seed or NetConfig()
    if not state or not state["network"]:
        return seed or NetConfig()
    try:
        return NetConfig(**state["network"])
    except TypeError:
        return seed or NetConfig()


def save_hosts(hosts: list[HostConfig], path: Path = HOSTS_JSON,
               net: NetConfig | None = None) -> None:
    """保存主机列表（带当前网络配置，避免两次写盘互相覆盖）。"""
    if net is None:
        net = load_net(path)
    _write_state(hosts, net, path)


def save_net(net: NetConfig, path: Path = HOSTS_JSON,
             hosts: list[HostConfig] | None = None) -> None:
    if hosts is None:
        hosts = load_hosts(path)
    _write_state(hosts, net, path)


def hosts_from_dicts(items: list[dict]) -> list[HostConfig]:
    """UI 表单/State 里的 dict 列表 → HostConfig（跳过无名行，校验必填）。"""
    out: list[HostConfig] = []
    seen: set[str] = set()
    for i, d in enumerate(items or []):
        name = str((d or {}).get("name") or "").strip()
        ssh = str((d or {}).get("ssh") or "").strip()
        # mode 缺失时按有没有 ssh 目标推断（旧的 hosts.json / config.yaml 没有这个字段，
        # 里面那台带 user@host 的机器必须还是 SSH 远程，否则命令会在 UI 这台机器上跑）
        mode = str((d or {}).get("mode") or "").strip()
        if mode not in (MODE_LOCAL, MODE_REMOTE):
            mode = MODE_REMOTE if ssh else MODE_LOCAL
        if not name and not ssh and mode == MODE_LOCAL and not (d or {}).get("remote_repo"):
            continue                   # 空行
        if not name:
            raise ValueError(f"第 {i + 1} 台机器：名称不能为空")
        if mode == MODE_REMOTE:
            if not ssh:
                raise ValueError(f"主机 {name}：SSH 机器的 user@host 不能为空")
            if "@" not in ssh:
                raise ValueError(f"主机 {name}：ssh 需要写成 user@host（当前 {ssh!r}）")
        else:
            ssh = ""                   # 本机不需要 ssh 目标
        if name in seen:
            raise ValueError(f"主机名重复：{name}")
        seen.add(name)
        try:
            port = int(str((d or {}).get("ssh_port") or 22).strip())
        except ValueError:
            raise ValueError(
                f"主机 {name}：端口必须是数字（当前 {(d or {}).get('ssh_port')!r}）")
        if not 0 < port < 65536:
            raise ValueError(f"主机 {name}：端口超出范围：{port}")

        def s(key: str, fallback: str) -> str:
            v = str((d or {}).get(key) or "").strip()
            return v or fallback

        out.append(HostConfig(
            name=name, ssh=ssh, mode=mode, ssh_port=port,
            password=str((d or {}).get("password") or ""),
            remote_repo=s("remote_repo", DEFAULT_REPO),
            venv_dir=s("venv_dir", DEFAULT_VENV),
            task_dir=s("task_dir", DEFAULT_TASK_DIR),
            hf_cache=s("hf_cache", DEFAULT_HF_CACHE),
        ))
    if not out:
        raise ValueError("至少需要一台目标机器")
    return out


# ---------------------------------------------------------------- yaml

def _seed_net(net_cfg: dict) -> NetConfig:
    """config.yaml 的 network 段 → NetConfig（只播种初值，之后以 UI 存的为准）。

    新式写 `mode` / `remote_proxy` / `local_proxy`；旧式 `remote_proxy: null | local_socks | "<url>"`
    不改也能跑。代理地址不在仓库里写死：这里留空就读本机 env，换台机器不至于带着个连不通的地址。
    """
    local = (net_cfg.get("local_proxy") or net_cfg.get("local_socks_proxy")
             or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy") or "")
    remote = net_cfg.get("remote_proxy", PROXY_AUTO)
    mode = str(net_cfg.get("mode") or "").strip()
    old = (None, "None", "local_socks", PROXY_AUTO)      # 旧语义：不是真地址
    if mode not in (PROXY_AUTO, PROXY_MANUAL, PROXY_DIRECT):
        if remote is None:
            mode = PROXY_DIRECT
        elif str(remote) in ("local_socks", PROXY_AUTO):
            mode = PROXY_AUTO
        else:
            mode = PROXY_MANUAL
    url = "" if str(remote) in (None, "None", "local_socks", PROXY_AUTO) else str(remote)
    return NetConfig(mode=mode, remote_proxy=url, local_proxy=str(local))


def load_config(path: str | Path | None = None) -> AppConfig:
    p = Path(path) if path else CONFIG_DIR / "config.yaml"
    raw: dict = {}
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}

    srv = raw.get("server", {}) or {}
    net_cfg = raw.get("network", {}) or {}
    return AppConfig(
        server_host=srv.get("host", "0.0.0.0"),
        server_port=int(srv.get("port", 7861)),
        hosts=load_hosts(),
        net=load_net(seed=_seed_net(net_cfg)),
        defaults=raw.get("defaults", {}) or {},
    )

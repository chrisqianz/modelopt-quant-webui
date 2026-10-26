"""配置加载。

两层来源：
- config.yaml：server 监听 + local 路径四件套 + defaults（可选，缺省即默认值）；
  network 段只作代理配置的初始种子。
- net.json：UI「网络」面板保存后的代理配置，存在则覆盖 yaml 种子。

代理地址一律手填，程序不猜拓扑。WebUI 跑在有 GPU 的这台机器上，路径都按本机解释。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent

# UI 维护的网络代理配置持久化位置（与 config.yaml 同目录，gitignore 掉）
NET_JSON = CONFIG_DIR / "net.json"

PROJECT_ROOT = CONFIG_DIR          # config.yaml 所在目录就是项目根
DEFAULT_REPO = "~/Model-Optimizer"
DEFAULT_VENV = ".venvs/modelopt"   # 相对项目根（ADR 024 D9）
DEFAULT_TASK_DIR = "~/quant-webui-tasks"
DEFAULT_HF_CACHE = "~/.cache/huggingface"

# 代理模式：三种互斥取值，UI 下拉直接对应。地址一律手填，不写死。
PROXY_AUTO = "auto"        # 先探直连，不通则回落到手填的本机代理地址
PROXY_MANUAL = "manual"    # 直接用填的代理地址
PROXY_DIRECT = "direct"    # 不注入任何代理
PROXY_MODES = [
    ("自动（先试直连，不通再用代理）", PROXY_AUTO),
    ("强制走代理（用下面填的地址）", PROXY_MANUAL),
    ("不用代理（直连下载）", PROXY_DIRECT),
]


def resolve_relative(p: str | Path) -> Path:
    """相对路径按项目根解析，再 expanduser。绝对路径与 ~ 原样。"""
    s = str(p or "").strip()
    if not s:
        return Path("")
    q = Path(s).expanduser()
    return q if q.is_absolute() else (PROJECT_ROOT / q).resolve()


@dataclass
class LocalConfig:
    """这台机器上的四个路径。"""
    repo: str = DEFAULT_REPO
    venv: str = DEFAULT_VENV
    task_dir: str = DEFAULT_TASK_DIR
    hf_cache: str = DEFAULT_HF_CACHE

    @property
    def repo_path(self) -> Path: return resolve_relative(self.repo)
    @property
    def venv_path(self) -> Path: return resolve_relative(self.venv)
    @property
    def task_path(self) -> Path: return resolve_relative(self.task_dir)
    @property
    def hf_cache_path(self) -> Path: return resolve_relative(self.hf_cache)


@dataclass
class NetConfig:
    mode: str = PROXY_AUTO
    proxy: str = ""
    hf_token: str = ""   # HF 访问令牌：gated 数据集用；只存 net.json，不回显前端


@dataclass
class AppConfig:
    server_host: str = "0.0.0.0"
    server_port: int = 7861
    server_token: str = ""
    local: LocalConfig = field(default_factory=LocalConfig)
    net: NetConfig = field(default_factory=NetConfig)
    defaults: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 持久化

def _read_state(path: Path = NET_JSON) -> dict | None:
    """读 net.json，返回 {"network": {...}}；文件不存在返回 None。"""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):       # 文件坏了就用 config.yaml 的种子，别把 UI 卡死
        return None
    if isinstance(raw, dict):
        return {"network": raw.get("network")}
    raise ValueError("net.json 格式无法识别")


def _write_state(net: NetConfig, path: Path = NET_JSON) -> None:
    path.write_text(json.dumps({"network": asdict(net)}, ensure_ascii=False, indent=2))
    path.chmod(0o600)                  # 代理地址可能带 user:pass，不给别人读


def load_net(path: Path = NET_JSON, seed: NetConfig | None = None) -> NetConfig:
    """读网络代理配置；UI 没存过就用 seed（来自 config.yaml / env）。"""
    try:
        state = _read_state(path)
    except Exception:
        return seed or NetConfig()
    if not state or not state["network"]:
        return seed or NetConfig()
    try:
        return NetConfig(**state["network"])
    except TypeError:                  # 旧条目（remote_proxy/local_proxy）直接退回种子
        return seed or NetConfig()


def save_net(net: NetConfig, path: Path = NET_JSON) -> None:
    """保存代理配置（UI 存的覆盖 config.yaml 里的种子）。"""
    _write_state(net, path)


def save_local(local: LocalConfig) -> None:
    """把 local 段写回 config.yaml。

    注意：保存会抹掉 config.yaml 的注释（yaml 重写不保留注释），
    所以路径的说明写在 UI 文案里，不靠配置文件。"""
    p = CONFIG_DIR / "config.yaml"
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    raw = raw or {}
    raw["local"] = {"repo": local.repo, "venv": local.venv,
                    "task_dir": local.task_dir, "hf_cache": local.hf_cache}
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))


# ---------------------------------------------------------------- yaml

def _load_local(d: dict) -> LocalConfig:
    return LocalConfig(
        repo=str(d.get("repo") or DEFAULT_REPO),
        venv=str(d.get("venv") or DEFAULT_VENV),
        task_dir=str(d.get("task_dir") or DEFAULT_TASK_DIR),
        hf_cache=str(d.get("hf_cache") or DEFAULT_HF_CACHE),
    )


def _seed_net(d: dict) -> NetConfig:
    proxy = (d.get("proxy") or d.get("local_proxy")
             or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy") or "")
    mode = str(d.get("mode") or "").strip()
    if mode not in (PROXY_AUTO, PROXY_MANUAL, PROXY_DIRECT):
        mode = PROXY_DIRECT if not proxy else PROXY_AUTO
    return NetConfig(mode=mode, proxy=str(proxy))


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
        server_token=str(srv.get("token", "") or ""),
        local=_load_local(raw.get("local", {}) or {}),
        net=load_net(seed=_seed_net(net_cfg)),
        defaults=raw.get("defaults", {}) or {},
    )

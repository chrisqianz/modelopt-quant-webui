# Plan 1 · 后端去 SSH 化 + 部署通道

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** 把执行层从「SSH 远程」改成「WebUI 跑在 GPU 机上」，删掉全部 SSH/隧道/密码代码，venv 挪进项目内，并建立一条可复现的部署通道 —— 全程 Gradio UI 仍可用，量化命令串一字不差。

**Architecture:** 先给 `build_params` / `build_cmd` / `build_eval_script` / `_result_tables` 套上 golden 快照（改代码前锁死行为），再一次性拔掉 SSH：`config.py` 用 `LocalConfig`（`config.yaml` 的 `local:` 段）替掉 `HostConfig` + `hosts.json`，`runner.py` 删 16 个 SSH 符号并把 8 处 `is_local` 分支拉平成单路径，`vocab.py`/`validate.py` 的远程拉取换成本机文件系统读取，`app.py` 的连接面板换成「路径 + 代理」。最后写 `deploy.sh` 把项目 rsync 到 GPU 机，用旧 Gradio 版在真机上跑一遍作为对照组。

**Tech Stack:** Python 3.12（`.venv`）、PyYAML、Gradio 6.30（本计划内仍是 UI）、pytest（新增）、rsync/ssh。

**设计依据:** `docs/superpowers/specs/2026-10-14-fastapi-react-ui-design.md` §5.1、§5.2、D3、D9、D10。
**不在本计划:** FastAPI 层、jobs/SSE、React 前端、产物库/缓存列表（Plan 2/3/4）。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `tests/golden_adapter.py` | 新建 | 把「当前代码怎么拼命令」隔离成一层，重构时只改它 |
| `tests/test_golden_command.py` | 新建 | 量化/评测命令与表格的快照回归 |
| `tests/fixtures/lm_eval_sample.json` | 新建 | `_result_tables` 的输入样本 |
| `webui/config.py` | 重写 | `LocalConfig`（路径四件套）+ `NetConfig`（只剩本机代理）+ `AppConfig` |
| `config.yaml` | 改 | 新增 `local:` 段，`network` 去掉 `remote_proxy`，注释同步 |
| `webui/runner.py` | 改 | 删 SSH 符号，拉平分支，代理单路径，venv 相对项目根 |
| `webui/vocab.py` | 改 | 配方/数据集扫描走本机文件系统 |
| `webui/validate.py` | 改 | 产物目录读取走本机文件系统 |
| `app.py` | 改 | 连接面板重做、去掉 `host_dd`、文案去「远程」 |
| `deploy.sh` | 新建 | rsync 到 GPU 机（带 venv 排除） |
| `hosts.json` | 删除 | 连 `_SEED` 播种逻辑一起 |

---

## Task 0 — 建目录与本计划落盘

- [ ] `mkdir -p tests/fixtures docs/superpowers/plans`
- [ ] 确认本文件在 `docs/superpowers/plans/`（要入库，不放 gitignore 的 `.superpowers/`）
- [ ] Commit: `git add docs/superpowers/plans && git commit -m "docs: Plan 1 后端去 SSH 化 + 部署通道"`

---

## Task 1 — Golden 快照网（改任何代码之前）

**Files:** `tests/golden_adapter.py`、`tests/test_golden_command.py`、`tests/fixtures/lm_eval_sample.json`

- [ ] 装 pytest：`.venv/bin/pip install pytest` → `echo "pytest" >> requirements.txt`

- [ ] 写 `tests/fixtures/lm_eval_sample.json`（lm-eval 输出形状，够 `_result_tables` 用）：

```json
{
  "results": {
    "gpqa_diamond_cot_zeroshot": {"acc,none": 0.3214, "acc_stderr,none": 0.0213, "alias": "gpqa_diamond"},
    "mmlu": {"acc,none": 0.6812, "acc_stderr,none": 0.0068, "alias": "mmlu"},
    "ifeval": {"prompt_level_strict_acc,none": 0.7129, "inst_level_strict_acc,none": 0.8341, "alias": "ifeval"}
  }
}
```

- [ ] 写 `tests/golden_adapter.py` —— **这一层是唯一允许在后续 Task 里改的测试代码**：

```python
"""把「当前代码怎么拼命令」包一层，重构时只改这里，fixture 不动。

Task 5 拔掉 host 维度时，只改 assemble_quantize / assemble_eval 的实现，
golden/*.json 必须原样通过 —— 这是「行为没变」的唯一证据。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  import 无副作用：main() 在 __main__ 守卫里


def assemble_quantize(p: dict) -> str:
    """参数字典 → 要执行的命令串。"""
    host = app.CFG.get_host(app.CFG.host_names()[0])
    return app.build_remote_cmd(host, p)


def assemble_eval(quant: str, baseline: str, tasks: list[str],
                  tp: int, port: int, batch: int) -> str:
    host = app.CFG.get_host(app.CFG.host_names()[0])
    return app.build_eval_script(host, quant, baseline, tasks, tp, port, batch)


def assemble_tables(results: list[dict]) -> list[list[str]]:
    rows, _note = app._result_tables(results)
    return rows
```

- [ ] 写参数用例 `tests/cases.py`（**不含 host_name**，位置参数由 adapter 按 `INPUT_ORDER` 补齐）：

```python
"""量化参数用例。keys 用 INPUT_ORDER 里的名字，adapter 负责按位置展开。"""

QUANTIZE_CASES: dict[str, dict] = {
    "hf_id_default": {
        "model": "meta-llama/Llama-3.1-8B", "trust_remote_code": True,
        "qformat": "nvfp4", "recipe": "", "kv_cache_qformat": "nvfp4",
        "datasets": ["cnn_dailymail"], "sizes_names": ["cnn_dailymail"],
        "sizes": {"cnn_dailymail": 1024}, "custom_datasets": [],
        "calib_seq": 512, "calib_with_images": False, "export_path": "",
        "batch_size": 1, "device": "cuda", "attn_implementation": "sdpa",
        "use_fsdp2": False, "cpu_offload": False, "low_memory_mode": False,
        "gpu_max_mem_percentage": 0.0, "skip_generate": False,
    },
    "local_path_recipe_multi_dataset": {
        "model": "~/models/Qwen3-8B", "trust_remote_code": True,
        "qformat": "nvfp4", "recipe": "~/Model-Optimizer/modelopt_recipes/llm/nvfp4.yaml",
        "kv_cache_qformat": "int8",
        "datasets": ["cnn_dailymail", "open_platypus"],
        "sizes_names": ["cnn_dailymail", "open_platypus"],
        "sizes": {"cnn_dailymail": 512, "open_platypus": 256},
        "custom_datasets": [["mysite", 128]],
        "calib_seq": 2048, "calib_with_images": True,
        "export_path": "~/quant-webui-tasks/out/qwen3-nvfp4",
        "batch_size": 4, "device": "cuda:0", "attn_implementation": "flash_attention_2",
        "use_fsdp2": True, "cpu_offload": True, "low_memory_mode": True,
        "gpu_max_mem_percentage": 0.9, "skip_generate": True,
    },
    "recipe_dir_minimal": {
        "model": "hf-org/Tiny-124M", "trust_remote_code": False,
        "qformat": "nvfp4", "recipe": "~/recipes/dir/", "kv_cache_qformat": "none",
        "datasets": [], "sizes_names": [], "sizes": {}, "custom_datasets": [],
        "calib_seq": 0, "calib_with_images": False, "export_path": "",
        "batch_size": 0, "device": "cuda", "attn_implementation": "",
        "use_fsdp2": False, "cpu_offload": False, "low_memory_mode": False,
        "gpu_max_mem_percentage": 0.0, "skip_generate": False,
    },
}

EVAL_CASES: dict[str, dict] = {
    "quant_only": {
        "quant": "~/quant-webui-tasks/out/llama-nvfp4", "baseline": "",
        "tasks": ["gsm8k", "mmlu"], "tp": 1, "port": 8001, "batch": 8,
    },
    "with_baseline_tp2": {
        "quant": "/data/exports/q1", "baseline": "/data/models/base",
        "tasks": ["gpqa_diamond_cot_zeroshot", "ifeval"], "tp": 2, "port": 8000, "batch": 1,
    },
}
```

- [ ] adapter 里加用例 → 位置参数的换算（`sizes` dict 按 `sizes_names` 顺序展开成 `sizes_table` 行；`host_name` 插在最前），补进 `tests/golden_adapter.py`：

```python
def quantize_args(c: dict) -> list:
    """用例 dict → gradio 事件的位置参数列表（a[0]=host_name，其余按 INPUT_ORDER）。"""
    names = c["sizes_names"]
    table = [[c["sizes"][n]] for n in names]
    row = {**c, "sizes_table": table}
    host_name = app.CFG.host_names()[0]
    return [host_name] + [row[k] for k in app.INPUT_ORDER if k in row]
```

> `INPUT_ORDER` 里的名字与用例 keys 必须一一对上；跑第一条测试前用 `print(app.INPUT_ORDER)` 核一遍，缺哪个补哪个（用例少 key 会让 `build_params` 取到 `None`，命令串就变了）。

- [ ] 写 `tests/test_golden_command.py`：

```python
import json
from pathlib import Path

import pytest

from tests import golden_adapter as ad
from tests.cases import EVAL_CASES, QUANTIZE_CASES

GOLDEN = Path(__file__).parent / "golden"
FIX = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((GOLDEN / f"{name}.json").read_text())


def _dump(name: str, obj) -> None:
    GOLDEN.mkdir(exist_ok=True)
    (GOLDEN / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture
def update(request):
    return request.config.getoption("--update-golden")


def test_quantize_cmd(update):
    want = {k: ad.assemble_quantize(ad.quantize_args(v)) for k, v in QUANTIZE_CASES.items()}
    if update:
        _dump("quantize_cmd", want)
    assert want == _load("quantize_cmd")


def test_eval_script(update):
    want = {k: ad.assemble_eval(**v) for k, v in EVAL_CASES.items()}
    if update:
        _dump("eval_script", want)
    assert want == _load("eval_script")


def test_result_tables(update):
    sample = json.loads((FIX / "lm_eval_sample.json").read_text())
    want = ad.assemble_tables([{"file": "quant", "data": sample},
                              {"file": "baseline", "data": sample}])
    if update:
        _dump("result_tables", want)
    assert want == _load("result_tables")
```

- [ ] 加 `--update-golden` 选项，写 `tests/conftest.py`：

```python
def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="重写 golden/*.json（只在故意改行为时用，改完看 diff）")
```

- [ ] 加空 `tests/__init__.py`（让 `from tests import ...` 成立）

- [ ] 生成快照并**人眼过一遍**命令串：

```bash
.venv/bin/pytest tests/test_golden_command.py --update-golden -q
cat tests/golden/quantize_cmd.json
```

Expected: 三条命令都以 `cd <repo> && <venv>/bin/python examples/hf_ptq/hf_ptq.py` 开头；用例 2 含 `--recipe` 而不是 `--qformat`、含 `--dataset 'cnn_dailymail,open_platypus,mysite' --calib_size '512,256,128'`、含 `--use_fsdp2 --cpu_offload --low_memory_mode`、`--gpu_max_mem_percentage 0.9`；用例 3 无 `--kv_cache_qformat`。
若与预期不符，**先查是不是 `INPUT_ORDER` 对齐错了**，不要改 `app.py`。

- [ ] 再跑一次（不带 flag）确认稳定：`.venv/bin/pytest -q` → Expected `3 passed`
- [ ] Commit: `git add tests requirements.txt && git commit -m "test: 锁死量化/评测命令串的 golden 快照（改代码前的安全网）"`

---

## Task 2 — config.py：`HostConfig` → `LocalConfig`

**Files:** `webui/config.py`（重写）、`config.yaml`、`hosts.json`（删）

- [ ] `config.yaml` 全文替换为（`local:` 新增、`network` 去掉 `remote_proxy`、注释不再提 SSH）：

```yaml
# ModelOpt 量化 WebUI 配置
# WebUI 跑在有 GPU 的这台机器上，路径都按「本机」解释。
# 换服务器：改下面四行路径 + 重启即可，不需要任何配置迁移。

server:
  host: 0.0.0.0          # 局域网内其他设备也能打开；只要本机访问改回 127.0.0.1
  port: 7861
  token: ""              # 非空则所有 /api/* 要求 X-Api-Token；局域网内留空即可

# 路径。相对路径按「config.yaml 所在目录」解析 —— venv 在项目内，跟着仓库走。
local:
  repo: ~/Model-Optimizer        # Model-Optimizer 仓库（hf_ptq.py 在 examples/hf_ptq/ 下）
  venv: .venvs/modelopt          # 量化用的虚拟环境（项目内，不进 ~）
  task_dir: ~/quant-webui-tasks  # 产物与日志；故意在仓库外，免得被 rsync 拖走
  hf_cache: ~/.cache/huggingface

# HuggingFace 下载走哪条网。代理地址一律手填，程序不猜拓扑。
# mode: auto 先试直连 huggingface.co，不通再用 proxy；manual 强制；direct 不用
network:
  mode: auto
  proxy: ""              # 例：http://127.0.0.1:7890

defaults:
  calib_size: 1024
  calib_seq: 512
  qformat: nvfp4
  kv_cache_qformat: none
```

- [ ] `webui/config.py` 重写。删掉：`HOSTS_JSON`、`MODE_LOCAL`/`MODE_REMOTE`/`EXEC_MODES`、`HostConfig`、`_SEED`、`load_hosts`、`save_hosts`、`hosts_from_dicts`、`NetConfig.remote_proxy`。新增/改动如下（`PROXY_*`、`_read_state`、`_write_state`、`load_net`、`save_net` 保留，`_write_state` 的 `chmod 600` 保留）：

```python
PROJECT_ROOT = CONFIG_DIR          # config.yaml 所在目录就是项目根
DEFAULT_REPO = "~/Model-Optimizer"
DEFAULT_VENV = ".venvs/modelopt"   # 相对项目根（ADR 024 D9）
DEFAULT_TASK_DIR = "~/quant-webui-tasks"
DEFAULT_HF_CACHE = "~/.cache/huggingface"


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


@dataclass
class AppConfig:
    server_host: str = "0.0.0.0"
    server_port: int = 7861
    server_token: str = ""
    local: LocalConfig = field(default_factory=LocalConfig)
    net: NetConfig = field(default_factory=NetConfig)
    defaults: dict = field(default_factory=dict)
```

`load_config` 里 `hosts=load_hosts()` → `local=_load_local(raw.get("local", {}) or {})`，`server_token=str(srv.get("token", "") or "")`：

```python
def _load_local(d: dict) -> LocalConfig:
    return LocalConfig(
        repo=str(d.get("repo") or DEFAULT_REPO),
        venv=str(d.get("venv") or DEFAULT_VENV),
        task_dir=str(d.get("task_dir") or DEFAULT_TASK_DIR),
        hf_cache=str(d.get("hf_cache") or DEFAULT_HF_CACHE),
    )
```

`_seed_net` 简化（不再兼容 `remote_proxy` 的 `local_socks` 旧语义）：

```python
def _seed_net(d: dict) -> NetConfig:
    proxy = (d.get("proxy") or d.get("local_proxy")
             or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy") or "")
    mode = str(d.get("mode") or "").strip()
    if mode not in PROXY_MODES:
        mode = PROXY_DIRECT if not proxy else PROXY_AUTO
    return NetConfig(mode=mode, proxy=str(proxy))
```

- [ ] 删文件：`git rm -f hosts.json`（若本地不存在则 `rm -f hosts.json`）
- [ ] 冒烟：`.venv/bin/python -c "from webui.config import load_config, resolve_relative as r; c=load_config(); print(c.server_port, c.local.venv_path, c.net.mode, repr(c.net.proxy))"`
      Expected: 第二项是 `/Users/chrisqian/Development/modelopt-quant-webui/.venvs/modelopt`（绝对、项目内），不含 `~`
- [ ] Commit: `git commit -am "refactor(config): HostConfig/hosts.json → LocalConfig/config.yaml local 段，venv 挪进项目内"`

> Task 2–5 之间 `app.py` 会 import 失败（引用了删掉的 `HostConfig` 等），属预期，**不要在中间提交**。

---

## Task 3 — runner.py 拔掉 SSH

**Files:** `webui/runner.py`

删除清单（模块级常量 + 函数，行号是删除前的位置）：

| 删 | 位置 |
|---|---|
| `SSH_OPTS`、`SSHPASS_HINT` | L24、L33 |
| `_REMOTE_HOME`、`_TUNNELS`、`_HF_DIRECT` | L64、L66、L67 |
| `_MASTER_OK`、`_CTL_NAME_LEN`、`_CTL_DIR` | L102–104 |
| `_host_key`、`_ssh_error`、`_is_conn_err` | L70、L74、L91 |
| `_ctl_dir`、`control_path`、`sshpass_bin`、`_port_args` | L107、L131、L137、L147 |
| `_ssh_base`、`master_alive`、`ensure_master`、`drop_master` | L151、L162、L170、L206 |
| `tunnel_scheme`、`_remote_port_open`、`ensure_hf_tunnel` | L441、L447、L455 |
| `remote_dir_summary`（0 引用） | L763 |
| `_HF_DIRECT` 相关：`hf_direct_ok` | L346（改为模块级 `_HF_OK` 单次缓存，或直接每次真测） |

改动清单：

- [ ] 顶部 import：删 `getpass`、`hashlib`、`tempfile`（只被 SSH 段用；`shlex`/`os`/`time`/`json`/`re`/`subprocess` 保留），docstring 去掉「远程 SSH 执行器」
- [ ] `q_remote(path, host=None)`（L37）→ 签名去 host，实现就是展开 + 转义：

```python
def q_remote(path: str | Path) -> str:
    """展开 ~ / 项目内相对路径再转义。存在理由：shlex.quote 会把 ~ 括进单引号，shell 不展开。"""
    p = resolve_relative(path)
    return shlex.quote(str(p.expanduser()))
```

- [ ] `remote_home(host)`（L53）→ 删掉，调用点全部换成 `q_remote` 或直接 `os.path.expanduser("~")`
- [ ] `_exec`（L244）保留原样 —— 它是唯一的进程执行原语
- [ ] `run_remote(host, cmd, timeout)`（L253）→ 改名 `run_shell(cmd, timeout)`，body 一行：

```python
def run_shell(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """在本机跑一段 bash。返回 (rc, stdout, stderr)。"""
    return _exec(["bash", "-c", cmd], timeout=timeout)
```

调用点全量替换（`app.py` + `webui/*.py` 共 40 处）：`run_remote(` → `run_shell(`，并把首参 `host,` 去掉。
- [ ] `run_local`（L275，只有 SSH 路径在用）→ 删
- [ ] `check_connection`（L218）→ 只留本机分支，语义变成「shell 可用 + 主机名」：

```python
def check_connection() -> str:
    rc, out, err = run_shell("echo __OK__; hostname; echo HOME=$HOME")
    if rc != 0 or "__OK__" not in out:
        return f"✗ 本机 shell 不可用：{(err or out)[:200]}"
    body = out.replace("__OK__", "").strip()
    return f"✓ 本机执行可用 · {body}"
```

- [ ] `HostStatus`（L283）删 `auth` 字段；`as_dict()` 里对应中文键「认证」一并删（`reachable/gpu/gpu_util/gpu_mem/gpu_busy/repo_ok/venv_ok/uv_ok/disk_free/hf_cache/hf_direct_ok/hf_proxy_ok/proxy_note/error` 全留）
- [ ] `probe_hf`（L326）：`run_remote(host, ...)` → `run_shell(...)`；错误文案 `return _ssh_error(err)` → `return f"✗ 探测失败：{(err or out)[:200]}"`
- [ ] `probe_host(host, app_cfg)`（L356）→ `probe_host(cfg: AppConfig)`：
  - 开头 `reachable` 走 `check_connection()`；删 `st.auth = host.auth_kind`
  - 路径判断：`test -d {repo}/examples/hf_ptq` 的 `repo` 用 `q_remote(cfg.local.repo)`；venv 用 `cfg.local.venv`；`df -h` 与 `du -sh` 用 `cfg.local.hf_cache`
  - 删 `ensure_hf_tunnel` / `_TUNNELS` 那段（AUTO 分支），改成：`mode == PROXY_AUTO` 且 `cfg.net.proxy` 非空 → 探测代理；为空则 `proxy_note = "未配置代理地址（设置里填）"`
- [ ] `proxy_url_for(app_cfg, host)`（L483）→ `proxy_url_for(cfg: AppConfig) -> tuple[str, str]`，只剩一条路：

```python
def proxy_url_for(cfg: AppConfig) -> tuple[str, str]:
    """→ (proxy url, 说明)。只有本机代理一条路，地址一律手填。"""
    if cfg.net.mode == PROXY_DIRECT:
        return "", "按设置不用代理"
    url = normalize_proxy(cfg.net.proxy)
    if not url:
        return "", "没填代理地址"
    if cfg.net.mode == PROXY_MANUAL:
        return url, f"强制走 {url}"
    if hf_direct_ok(cfg):
        return "", "直连可用，没用代理"
    return url, f"直连不通，走 {url}"
```

- [ ] `proxy_for`（L509）、`test_proxy`（L515）同步去 host；`test_proxy` 三模式各自 `probe_hf` 的结构保留，`host` 参数删
- [ ] `ensure_venv`（L554）/ `ensure_eval_env`（L585）：命令里的 `{venv}` 换成 `resolve_relative(cfg.local.venv)`，其余（`uv venv … --python 3.12` + `VIRTUAL_ENV=… uv pip install -e '.[hf]'`、marker `VENV_DONE`、评测环境钉 `'vllm>=0.10.1' 'lm_eval[ifeval]>=0.4.12,<0.5'` + marker `EVAL_ENV_DONE`）一字不动
- [ ] `Task`（L679）删 `host` 字段；`start_task`（L692）、`start_script_task`（L712）、`poll_task`（L743）、`read_log`（L752）、`remote_tail`（L756）、`abort_task`（L777）、`export_file_listing`（L786）去掉 `host` 形参，内部 `run_remote(host, …)` → `run_shell(…)`。`remote_tail` 改名 `tail_log(path, tail_bytes)`
- [ ] `_proxy_env(proxy)`（L541）、`normalize_proxy`（L419）、`_parse_proxy_addr`（L433）、`_resolve_uv`（L549）、`_wait_for_marker`（L611）、`probe_eval_env`（L626）、`fetch_eval_results`（L642）逻辑保留，只去 host 参数
- [ ] 语法与残留检查：

```bash
.venv/bin/python -m py_compile webui/runner.py
grep -nE "ssh|sshpass|ControlMaster|is_local|remote_repo|_TUNNELS|_MASTER_OK|run_remote|MODE_REMOTE" webui/runner.py
```

Expected: `grep` 只剩注释里的历史说明（应该 0 命中最好；命中的每一行都要么改名要么加一句「历史：原 SSH 时代」再确认读不出任何 SSH 调用）
- [ ] Commit: `git commit -am "refactor(runner): 删除 SSH/ControlMaster/反向隧道，执行层拉平成单路径本机执行"`

---

## Task 4 — vocab.py / validate.py 改本机读取

**Files:** `webui/vocab.py`、`webui/validate.py`

- [ ] `vocab.fetch_remote_qformats(host)`（L112-134）→ `scan_qformats(repo: Path) -> list[str]`：同一套解析（`grep -oE "choices=\[[^]]*\]"` + `nvfp4|fp8|int8` 过滤）改成 `Path.read_text()` + `re.findall`，源文件 `repo/modelopt/torch/quantization/config.py`；读不到时回落到现有内置快照（函数里那份 fallback 列表保留）
- [ ] `vocab.fetch_remote_datasets(host)`（L137-160）→ `scan_datasets(repo: Path)`，同上
- [ ] `vocab.render_qformat_dropdown`（L163）调用点同步；`all_qformats`（L99）、`kv_choices`（L103）、`builtin_dataset_keys`（L59）、`dataset_label`（L63）不动
- [ ] `validate.fetch_remote_export`（L54-75）→ `scan_export_dir(path: str) -> dict`：`os.walk(resolve_relative(path))`，产出结构与原来一致（文件清单 + `hf_quant_config.json` 解析），日志目录侧仍从产物目录读
- [ ] 冒烟：

```bash
.venv/bin/python -m py_compile webui/vocab.py webui/validate.py
.venv/bin/python -c "
from webui.config import load_config
from webui import vocab
c = load_config()
qs = vocab.scan_qformats(c.local.repo_path)
print(len(qs), qs[:5])"
```

Expected: 在 mac 上若没 `~/Model-Optimizer` → 返回内置快照（len > 0）；在 GPU 机上跑应返回真实扫到的列表（含 `nvfp4`）
- [ ] Commit: `git commit -am "refactor(vocab,validate): 远程 grep/cat 换成本机文件系统扫描"`

---

## Task 5 — app.py：连接面板重做

**Files:** `app.py`

- [ ] 顶部 import 改为只引 `DEFAULT_HF_CACHE, DEFAULT_REPO, DEFAULT_TASK_DIR, DEFAULT_VENV, PROXY_AUTO, PROXY_DIRECT, PROXY_MANUAL, PROXY_MODES, LocalConfig, NetConfig, load_config, save_net`，删 `EXEC_MODES, MODE_LOCAL, MODE_REMOTE, HostConfig, hosts_from_dicts, save_hosts`
- [ ] docstring L7-8「既能从开发机 SSH 驱动 GPU 机器」改成「跑在 GPU 机器上，浏览器从局域网打开」
- [ ] 删 SSH 相关辅助：`_host_rows`（L471）、`_host_list_md`（L475）、`_host_form`（L485）、`_apply_hosts`（L499）
- [ ] `pane_conn`（L717-790）整块替换为「运行环境」+「HuggingFace 网络」两列：

```python
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
                                venv_btn = gr.Button("创建 venv", size="sm")
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
```

- [ ] 事件绑定（放在原 host 事件的位置，替换掉 `pick_host`/`new_host`/`toggle_ssh`/`save_host`/`del_host`/`test_conn`/`reconnect` 那批）：

```python
        def do_save_env(repo, venv, task, cache):
            CFG.local.repo, CFG.local.venv = repo.strip() or DEFAULT_REPO, venv.strip() or DEFAULT_VENV
            CFG.local.task_dir = task.strip() or DEFAULT_TASK_DIR
            CFG.local.hf_cache = cache.strip() or DEFAULT_HF_CACHE
            save_local(CFG.local)
            st = runner.probe_host(CFG)
            return (_env_note_md(st), *_probe_out(st))

        save_env_btn.click(do_save_env, [ed_repo, ed_venv, ed_task, ed_cache],
                           [env_note, *PROBE_OUTS])

        def do_save_net(mode, proxy):
            CFG.net = NetConfig(mode=mode, proxy=proxy.strip())
            save_net(CFG.net)
            return runner.test_proxy(CFG)

        save_net_btn.click(do_save_net, [net_mode, net_proxy], [net_note])
        test_net_btn.click(lambda: runner.test_proxy(CFG), None, [net_note])
```

- [ ] `config.py` 补 `save_local(local: LocalConfig)`：写回 `config.yaml` 的 `local:` 段 —— 用 `yaml.safe_load` 读原文 + 改 dict + `yaml.safe_dump(..., allow_unicode=True, sort_keys=False)`，**保留注释会丢**，所以在函数 docstring 明说「保存会抹掉 config.yaml 的注释」，并把 `local:` 段的注释改成写在 UI 文案里（已做）：

```python
def save_local(local: LocalConfig) -> None:
    """把 local 段写回 config.yaml。注意：重写会丢掉文件里的注释。"""
    p = CONFIG_DIR / "config.yaml"
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    raw = raw or {}
    raw["local"] = {"repo": local.repo, "venv": local.venv,
                    "task_dir": local.task_dir, "hf_cache": local.hf_cache}
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
```

- [ ] 全仓改名（`app.py` 为主）：`runner.run_remote(host, x, timeout=n)` → `runner.run_shell(x, timeout=n)`；`build_remote_cmd(host, p)` → `build_cmd(p)`；`build_eval_script(host, ...)` → `build_eval_script(...)`；`host_dd` 及其 `gr.Dropdown` 定义删掉，`all_inputs`（L806-808）去掉首项 `host_dd`，`INPUT_ORDER`（L31-38）去掉 `host_name`
- [ ] `check_local_model(host, model)`（L114）→ `check_local_model(model)`，错误文案去掉「（本机模式=这台，远程模式=远程机）」，改成 `f"没有找到这个目录：{model}\n路径写在 WebUI 所在这台机器上。"`
- [ ] 全仓文案清「远程」：`grep -n "远程\|SSH\|隧道\|密码" app.py webui/*.py` 逐条改（`_hf_status_text` L318 的「✓ 远程直连可达」→「✓ 直连可达」）
- [ ] golden 只改 adapter（`tests/golden_adapter.py`），fixture 不动：

```python
def assemble_quantize(p: dict) -> str:
    return app.build_cmd(p)


def assemble_eval(quant, baseline, tasks, tp, port, batch) -> str:
    return app.build_eval_script(quant, baseline, tasks, tp, port, batch)


def quantize_args(c: dict) -> list:
    names = c["sizes_names"]
    row = {**c, "sizes_table": [[c["sizes"][n]] for n in names]}
    return [row[k] for k in app.INPUT_ORDER if k in row]
```

- [ ] 全绿验证：

```bash
.venv/bin/python -m py_compile app.py webui/*.py
.venv/bin/pytest -q
.venv/bin/python -c "import app; print('import ok')"
```

Expected: `3 passed`，`import ok`
- [ ] 本地起 UI 点一遍（mac，无 GPU，探测会红）：`.venv/bin/python app.py` → 打开 `http://127.0.0.1:7861`，确认：量化区渲染、预演能出命令串、连接设置区显示新面板、无 traceback
- [ ] Commit: `git commit -am "refactor(app): 连接面板改「运行环境 + HF 网络」，去掉机器 CRUD/执行位置/密码框"`

---

## Task 6 — 部署通道 deploy.sh

**Files:** `deploy.sh`

- [ ] 写 `deploy.sh`（目标机可用环境变量覆盖；`--exclude='.venvs/'` 是硬要求，漏了 `--delete` 会把 GPU 机上装好的 venv 删掉）：

```bash
#!/usr/bin/env bash
# 部署到 GPU 机。用法：./deploy.sh [user@host:path]
set -euo pipefail
DEST="${1:-chrisqian@192.168.124.39:~/modelopt-quant-webui/}"
cd "$(dirname "$0")"

rsync -avz --delete \
    --exclude='.git/' --exclude='.venv/' --exclude='.venvs/' \
    --exclude='__pycache__/' --exclude='web/node_modules/' \
    --exclude='web/dist/' --exclude='.lens/' --exclude='.superpowers/' \
    --exclude='hosts.json' --exclude='server.json' --exclude='*.log' \
    ./ "$DEST"

echo "部署完成 → $DEST"
echo "在目标机上：ssh \$(echo $DEST | cut -d: -f1) 'cd modelopt-quant-webui && .venv/bin/python app.py'"
```

- [ ] `chmod +x deploy.sh`
- [ ] `.gitignore` 补一行 `.venvs/`（项目内 venv 目录名，与 `.venv/` 不同），若还没有：`grep -qx '.venvs/' .gitignore || echo '.venvs/' >> .gitignore`
- [ ] Commit: `git commit -am "build: deploy.sh（rsync 部署通道，排除项目内 venv 与产物）"`

---

## Task 7 — GPU 机落地 + 端到端验证

**目标机:** `chrisqian@192.168.124.39`（`chrisqian-ubuntu01`，RTX 5090 D 32GB，ModelOpt `0.48.0rc0-35-g254195350` 在 `~/Model-Optimizer`，无 `~/.venvs/modelopt`，node v20.20.2）

- [ ] 部署：`./deploy.sh`
- [ ] 建 GPU 机上的运行 venv（**跑 UI 用**，与量化用的 `.venvs/modelopt` 是两回事）：

```bash
ssh chrisqian@192.168.124.39 'cd modelopt-quant-webui && python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt && .venv/bin/python -c "import gradio; print(gradio.__version__)"'
```

Expected: 打印 `6.30.0`（或 requirements 允许的版本）
- [ ] 起 UI 并确认监听：

```bash
ssh chrisqian@192.168.124.39 'cd modelopt-quant-webui && nohup .venv/bin/python app.py >/tmp/moweb.log 2>&1 & sleep 12; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7861/'
```

Expected: `200`；`/tmp/moweb.log` 无 traceback
- [ ] 浏览器（mac 上）打开 `http://192.168.124.39:7861`，逐项确认：
  - 显存/探活卡片显示真实 GPU（RTX 5090 D + 占用）
  - 连接设置里 venv 路径是 `.../modelopt-quant-webui/.venvs/modelopt`
  - 代理填 `http://192.168.124.24:7890`、模式 manual → 「测试连通性」返回 2xx/204 + 耗时
  - 词表下拉是真扫出来的（配方来自 `~/Model-Optimizer/modelopt_recipes`）
- [ ] 建量化 venv（一次性，5–15 分钟，会占网络）：UI 点「创建 venv」，日志尾部出现 `VENV_DONE`
- [ ] **预演**：填 `~/Model-Optimizer` 里能拿到的最小模型（或缓存里已有的小模型），点「预演命令」，把命令串与 golden 里同参数用例逐段对齐（路径不同是预期的）
- [ ] 真跑（**只在显存有余量时**）：先 `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` —— 若 `used > 24000MiB` 就**跳过这步**并在计划里记「显存被占，真跑留到 Task 8」，不要抢正在用的 GPU（ADR 023 不抢占约束）。有余量时跑一个 calib_size=16、单数据集的小样，确认：日志解析出阶段进度、产物目录出现 `hf_quant_config.json`、`对配方` 面板能出 diff
- [ ] Commit: `git commit -am "docs: GPU 机落地验证记录（deploy + 探测 + 代理 + 词表）"`

---## Task 8 — 收尾与文档

- [ ] 更新 `README.md`：删 SSH/hosts.json 段，加「跑在 GPU 机上 + `./deploy.sh`」，路径配置说明指向 `config.yaml`
- [ ] `AGENTS.md`（若存在）同步：执行模型 = 本机
- [ ] 残留终审，期望全部 0 命中：

```bash
grep -rniE "sshpass|ControlMaster|ssh_port|MODE_REMOTE|hosts\.json|HostConfig|fetch_remote|remote_repo|is_local" \
  app.py webui/ config.yaml README.md tests/ || echo CLEAN
```

- [ ] 全量回归：`.venv/bin/pytest -q && .venv/bin/python -c "import app"`
- [ ] 记忆同步（Pi-Memory 写盘门禁）：`03-Decisions.md` ADR 024 追加「已实现于 Plan 1」一行；`05-Active-Context.md` 更新进度（≤6144B）
- [ ] Commit: `git commit -am "docs: Plan 1 收尾 —— README/AGENTS 去 SSH、ADR 024 推进"`

---

## 自检（写计划的人自己先过一遍）

- **覆盖**：设计 §5.1 删除清单 → Task 3/4/5；§5.2 配置形态 + deploy.sh → Task 2/6；D9（venv 进项目）→ Task 2/3；§2「Gradio 版先在 GPU 机跑通」→ Task 7。§5.3–§5.6、§6–§8 属 Plan 2/3/4，本计划不含，无遗漏。
- **placeholder 扫描**：Task 5 的 `PROBE_OUTS` / `_env_note_md` / `_probe_out` 是 `app.py` 里已有的探测输出（现名 `do_refresh` 的返回元组），执行时按现有 `do_refresh` 的实际返回变量名接上，**不要新造**；这是本计划唯一需要执行者对齐现有变量名的地方，已在步骤里标明。
- **类型一致**：`run_shell(cmd, timeout)`、`probe_host(cfg)`、`proxy_url_for(cfg)`、`build_cmd(p)`、`scan_qformats(repo)`、`save_local(local)` 在 Task 2/3/4/5 之间引用一致；`resolve_relative` 定义在 `config.py`，`runner.py`/`validate.py` 从 `webui.config` import。
- **风险**：Task 2–5 中间不可运行（import 断），已明确禁止中间提交；Task 1 的 golden 是唯一行为证据，所以必须在任何删除之前完成。

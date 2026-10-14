# Plan 2 · FastAPI server + jobs/SSE + capabilities

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** 在现有 Gradio 入口（`app.py`，7861）**旁边**加一条 FastAPI 入口（`serve.py`，7862），把 27 个 handler 全部翻译成 22 个 REST + SSE 端点，引入 job registry + SSE + 落盘续传，并实现与 Model-Optimizer 的版本耦合三层应对（能力探测 / 发射前校验 / 附加参数逃生口）。本计划**不碰**量化命令本身（`build_cmd` 产出逐字节不变，golden 继续通过），**不写** React 前端（Plan 3），**不做** C1/C2 之外的新增功能。

**Architecture:** 五层，自底向上：
1. `webui/cmds.py`（新）：从 `app.py` 把 7 个 UI-agnostic 的纯函数搬出来（`build_params` / `build_cmd` / `build_eval_script` / `result_tables` / `check_local_model` / `task_slug` / `is_local_model` / `fmt_val` / `EVAL_PRESETS` / `EVAL_TASKS`），全部加 `cfg: AppConfig` 参数（不再依赖模块级 `CFG`）。`app.py` 改成 `from webui.cmds import ...`，legacy Gradio UI 行为一字不差。
2. `server/schemas.py`（新）：pydantic 请求/响应模型，直接吐 `ProgressReport` / `HostStatus` / `Task` 的 dataclass 字段，不再拼 HTML。
3. `server/jobs.py`（新）：job registry（`dict[job_id, Job]`）+ 后台 poller + SSE 扇出 + 落盘 `task_dir/<job_id>/job.json` + 启动时重挂 pid 存活的 job。
4. `server/capabilities.py`（新）：版本 + `hf_ptq.py` flag 静态抽取 + `FLAG_UI` 映射表 → `{version, commit, flags_all[], mapped[], unmapped[], missing[]}`。
5. `server/api.py`（新）：FastAPI `APIRouter`，22 个端点，一一对应现有 handler。`serve.py`（新）是入口：`uvicorn.run(server.api:app, host=CFG.server_host, port=CFG.server_port+1)`，默认 **7862**（legacy 在 7861，双开对照）。

**Tech Stack:** Python 3.10+（GPU 机 3.10.12 / mac `.venv` 3.12）、FastAPI、uvicorn、pydantic≥2、pytest + httpx（`TestClient`）。Gradio 6.30 保留（`app.py` 7861 仍跑，作 golden 来源与双开对照）。

**设计依据:** `docs/superpowers/specs/2026-10-14-fastapi-react-ui-design.md` §4（架构与目录）、§5.3（API 表 22 端点）、§5.4（jobs/SSE）、§5.5（schemas 直吐字段）、§5.6（`server.token`）、§7（版本耦合三层）、§9（验证策略）。
**不在本计划:** React 前端（Plan 3）、C1 产物库 / C2 缓存列表（Plan 4，本计划只实现 `/api/artifacts` 与 `/api/models/local` 的**最小版**——扫目录返回原始字段，不做卡片/vLLM 命令模板）、显存估算器（Plan 4）、删 `app.py`/`legacy_app_gradio.py`（Plan 4 收尾）。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `webui/cmds.py` | 新建 | 命令拼装 / 结果解析（UI-agnostic，`cfg` 参数化） |
| `app.py` | 改 | 删搬走的 7 个函数，改 import；其余（`_build_app`/`_prog_html`/`main`/`CFG`）不动 |
| `serve.py` | 新建 | FastAPI 入口（uvicorn，端口 7862） |
| `server/__init__.py` | 新建 | 包标记 |
| `server/schemas.py` | 新建 | pydantic 模型 + 错误 envelope |
| `server/jobs.py` | 新建 | registry + poller + SSE + 落盘/重挂 |
| `server/capabilities.py` | 新建 | 版本 + flag 抽取 + 映射表 |
| `server/api.py` | 新建 | 22 端点（`APIRouter`）+ token 中间件 |
| `server/static.py` | 新建 | `web/dist` 托管 + SPA fallback + 缺失时明确报错 |
| `tests/server/conftest.py` | 新建 | 共享 fixtures：临时 `AppConfig`、`client`（`TestClient`） |
| `tests/server/test_schemas.py` | 新建 | pydantic 模型 round-trip |
| `tests/server/test_capabilities.py` | 新建 | flag 抽取 / 映射表（对临时假仓库） |
| `tests/server/test_jobs.py` | 新建 | registry / 落盘 / 重挂 / SSE 扇出 |
| `tests/server/test_api_smoke.py` | 新建 | 全部端点冒烟（`TestClient`，mock 掉 shell） |
| `requirements.txt` | 改 | 加 `fastapi` / `uvicorn` / `pydantic>=2` / `httpx` |

---

## 全局约束

- **量化命令串逐字节不变**：`build_cmd` / `build_eval_script` / `result_tables` 搬进 `cmds.py` 后，`tests/golden/*.json` 必须原样通过（`--update-golden` 只在故意改行为时用）。
- **不碰 GPU**：mac `.venv` 无 CUDA，所有单测走 `TestClient` + mock 掉 `run_shell`；碰 GPU 的验证在 5090（`chrisqian@192.168.124.39`）上做。
- **不引入前端依赖**：`web/` 目录本计划不建（Plan 3 的事）。`static.py` 托管 `web/dist` 但目录不存在时明确报错「先 `npm run build`」，不白屏。
- **不改 `webui/runner.py` / `progress.py` / `vocab.py` / `validate.py` / `config.py` 的对外签名**：本计划只新增 `cmds.py`，其余业务逻辑原样复用。
- **`app.py` 的 7861 仍跑**：Plan 2 完成后双开对照（legacy Gradio 7861 + 新 FastAPI 7862），同份 `config.yaml`。
- **端口约定**：legacy `app.py` = `config.yaml` 的 `server.port`（7861），新 `serve.py` = `CFG.server_port + 1`（7862），**不新增配置键**。`deploy.sh` 不变（`web/` 不存在时 rsync 不报错）。

---

## Task 0 — 准备

- [ ] `mkdir -p server tests/server`
- [ ] `echo "fastapi" >> requirements.txt; echo "uvicorn" >> requirements.txt; echo "pydantic>=2" >> requirements.txt; echo "httpx" >> requirements.txt`
- [ ] `.venv/bin/pip install fastapi uvicorn "pydantic>=2" httpx` → 成功（GPU 机本任务不做，Task 7 部署时再装）
- [ ] `git add requirements.txt server/__init__.py tests/server docs/superpowers/plans/2026-10-14-plan2-fastapi-server.md && git commit -m "chore: Plan 2 准备 —— fastapi/uvicorn/pydantic 依赖 + 计划落盘"`（`server/__init__.py` 先写 `"""FastAPI server 包。"""`）

---

## Task 1 — 把命令拼装 / 结果解析搬进 `webui/cmds.py`

**Files:** `webui/cmds.py`（新建）、`app.py`（删搬走的 7 个函数 + 改 import）、`tests/golden_adapter.py`（改 import 路径）

**Interfaces:**
- Consumes: `webui.config.AppConfig`、`webui.runner.run_shell` / `q_remote`
- Produces（`webui/cmds.py` 对外签名，**全部带 `cfg` 参数**）：
  - `fmt_val(x) -> str`（不变）
  - `is_local_model(model: str) -> bool`（不变）
  - `task_slug(model: str) -> str`（不变）
  - `check_local_model(cfg: AppConfig, model: str) -> str`（**新增 cfg 参数**）
  - `build_params(a: list, cfg: AppConfig) -> dict`（**新增 cfg 参数**，`DEFAULTS` 改成 `cfg.defaults`）
  - `build_cmd(p: dict, cfg: AppConfig) -> str`（**新增 cfg 参数**，`CFG.local` → `cfg.local`）
  - `build_eval_script(quant: str, baseline: str, tasks: list[str], tp: int, port: int, batch: str, cfg: AppConfig) -> str`（**新增 cfg 参数**，当前 body 里没用到 CFG，但为一致性加上）
  - `result_tables(results: list[dict]) -> tuple[list[list[str]], str]`（不变，原 `_result_tables`）
  - 常量：`EVAL_PRESETS: dict[str, list[str]]`、`EVAL_TASKS: list[str]`（原样搬）

**Steps:**

- [ ] 新建 `webui/cmds.py`，把 `app.py:72-365` 的 `fmt_val` / `build_params` / `is_local_model` / `task_slug` / `check_local_model` / `build_cmd` / `build_eval_script` / `_result_tables` + 顶部 `EVAL_PRESETS` / `EVAL_TASKS` 常量全部搬进来。**每处原来读模块级 `CFG` 的地方改成读参数 `cfg`**（`CFG.local.repo` → `cfg.local.repo`、`DEFAULTS.get(...)` → `cfg.defaults.get(...)`）。`_result_tables` 改名为 `result_tables`（去掉前导下划线，对外公开）。**三处** `gr.Error`（`build_cmd` 内 1 处「请填写模型路径/名称」、`build_eval_script` 内 2 处「请填量化模型路径」/「请选评测任务」）全改成 `ValueError`（cmds 不该依赖 gradio；漏掉 `build_cmd` 那处会在搬完后仍 import gradio）。

- [ ] 改 `app.py`：
  - `from webui import progress, runner, validate, vocab` 后面加 `from webui.cmds import (EVAL_PRESETS, EVAL_TASKS, build_cmd, build_eval_script, build_params, check_local_model, fmt_val, is_local_model, result_tables, task_slug)`
  - 删掉 `app.py:56-365` 的 `EVAL_PRESETS` / `EVAL_TASKS` / `fmt_val` / `build_params` / `is_local_model` / `task_slug` / `check_local_model` / `build_cmd` / `build_eval_script` / `_result_tables` 的定义（**保留** `_prog_html` / `_prog_idle` / `_prog_text` / `_ev_progress` / `_env_line` / `_net_line` / `MO_CSS` / `_theme` / `main` / `_build_app` / `INPUT_ORDER`）。
  - `_build_app` 里所有调用点补上 `CFG` 参数：`build_params(list(a))` → `build_params(list(a), CFG)`、`build_cmd(build_params(list(a), CFG), CFG)`、`build_eval_script(..., CFG)`、`check_local_model(model)` → `check_local_model(CFG, model)`、`_result_tables(results)` → `result_tables(results)`。

- [ ] 改 `tests/golden_adapter.py`：`assemble_quantize` / `assemble_eval` / `assemble_tables` 的 `app.build_cmd` / `app.build_eval_script` / `app._result_tables` 改成 `webui.cmds.build_cmd` / `webui.cmds.build_eval_script` / `webui.cmds.result_tables`，并补上 `cfg` 参数（`app.CFG`）。`quantize_args` 不动（仍按 `app.INPUT_ORDER` 展开）。

- [ ] 验证：`.venv/bin/python -m pytest -q` → **3 passed**（golden 逐字节不变，证明搬函数没改行为）。

- [ ] 验证：`.venv/bin/python -c "import app; app._build_app(); print('LEGACY BUILD OK')"` → `LEGACY BUILD OK`（legacy Gradio UI 仍能构建）。

- [ ] Commit: `git add webui/cmds.py app.py tests/golden_adapter.py && git commit -m "refactor(cmds): 把 build_cmd/build_eval_script/result_tables 搬进 webui.cmds，cfg 参数化"`

---

## Task 2 — `server/schemas.py` + `server/__init__.py` + `serve.py` 骨架

**Files:** `server/schemas.py`、`serve.py`、`tests/server/test_schemas.py`

**Interfaces:**
- Produces（`server/schemas.py`）：
  - `class QuantizeParams(BaseModel)`：字段与 `build_params` 的 dict key 一致（`model: str`、`trust_remote_code: bool`、`qformat: str`、`recipe: str`、`kv_cache_qformat: str`、`datasets: list[str]`、`dataset_sizes: dict[str, int]`、`custom_datasets: list[list[str]]`、`calib_seq: int`、`calib_with_images: bool`、`export_path: str`、`batch_size: int`、`device: str`、`attn_implementation: str`、`use_fsdp2: bool`、`cpu_offload: bool`、`low_memory_mode: bool`、`gpu_max_mem_percentage: float`、`skip_generate: bool`、`extra_args: str = ""`——`extra_args` 是附加参数逃生口，**本计划只存不拼**，Plan 4 接入命令尾部）
  - `class PreviewRequest(BaseModel)`：`QuantizeParams`
  - `class EvalStart(BaseModel)`：`quant: str`、`baseline: str = ""`、`tp: int = 1`、`port: int = 8000`、`batch: str = "auto"`、`tasks: list[str]`
  - `class RecipeDiffRequest(BaseModel)`：`job_id: str`、`official: str = ""`（对照 legacy `do_validate`：输入是「任务」不是目录，日志里才有导出目录线索）
  - `class PathsRequest(BaseModel)`：`repo: str`、`venv: str`、`task_dir: str`、`hf_cache: str`
  - `class NetRequest(BaseModel)`：`mode: str`、`proxy: str = ""`
  - `class ErrorResponse(BaseModel)`：`error: dict`（`{code, message, detail}`）
- Produces（`serve.py`）：`def main() -> None`，`uvicorn.run("server.api:app", host=CFG.server_host, port=CFG.server_port + 1)`（默认 7862；`CFG.server_port` 读 `config.yaml` 的 `server.port`=7861，+1 即 7862）。

**Steps:**

- [ ] 写 `server/schemas.py`（pydantic v2，`model_config = ConfigDict(extra="forbid")` 防前端多传字段）。`QuantizeParams` 用 `model_validate` 把前端 JSON 转成 `build_params` 认的 dict（`to_build_params_dict(self) -> dict` 方法）。

- [ ] 写 `serve.py`：`from webui.config import load_config; CFG = load_config()`；`def main(): uvicorn.run("server.api:app", host=CFG.server_host, port=CFG.server_port + 1, log_level="info")`；`if __name__ == "__main__": main()`。**此时 `server.api` 还不存在**，所以本步先写一个占位 `server/api.py`（`from fastapi import FastAPI; app = FastAPI(title="ModelOpt Quant WebUI")` + 一个 `GET /api/health` 返回 `{"ok": true}`），让 `serve.py` 能起。

- [ ] 验证：`.venv/bin/python serve.py &` → `curl -s http://127.0.0.1:7862/api/health` → `{"ok":true}`；`kill %1`。

- [ ] 写 `tests/server/test_schemas.py`：`QuantizeParams(...).to_build_params_dict()` 字段齐全；`ErrorResponse` 能构造；`PathsRequest` / `NetRequest` 能解析。`.venv/bin/python -m pytest tests/server/test_schemas.py -q` → 全绿。

- [ ] Commit: `git add server/schemas.py serve.py server/api.py tests/server && git commit -m "feat(server): schemas + serve.py 骨架 + /api/health"`

---

## Task 3 — `server/capabilities.py`（版本 + flag 抽取 + 映射表）

**Files:** `server/capabilities.py`、`tests/server/test_capabilities.py`

**Interfaces:**
- Produces：
  - `FLAG_UI: dict[str, str]`：UI flag → `hf_ptq.py` flag 的映射表（18 项，值即 flag 名本身，如 `{"qformat": "--qformat", "kv_cache_qformat": "--kv_cache_qformat", "calib_seq": "--calib_seq", ...}`）。**这 18 项与 `build_cmd` 里实际拼的 flag 一致**（见 `webui/cmds.py` 的 `parts.append`）。
  - `extract_flags(repo: Path) -> list[str]`：对 `repo/examples/hf_ptq/hf_ptq.py` 做**静态** `re.findall(r"add_argument\(r?[\"'](--[a-zA-Z0-9_]+)", src)`，返回去重后的 flag 列表（**不跑** `--help`，那要 import torch）。
  - `probe_capabilities(cfg: AppConfig) -> dict`：`{version: str, commit: str, flags_all: list[str], mapped: list[str], unmapped: list[str], missing: list[str]}`。`version` / `commit` 用 `runner.run_shell(f"git -C {repo} describe --tags --always 2>/dev/null || echo unknown")` 与 `git -C {repo} rev-parse --short HEAD`；`mapped` = `FLAG_UI` 值 ∩ `flags_all`；`unmapped` = `flags_all` − `FLAG_UI` 值；`missing` = `FLAG_UI` 值 − `flags_all`。
  - `validate_cmd_flags(cmd: str, flags_all: list[str]) -> list[str]`：从命令串里抽 `--xxx` token，返回不在 `flags_all` 里的（用于发射前校验）。

**Steps:**

- [ ] 写 `server/capabilities.py`。`FLAG_UI` 的 18 项**必须**与 `webui/cmds.py` 的 `build_cmd` 里 `parts.append` 的 flag 一一对应（逐项对照，不靠记忆）：`--pyt_ckpt_path`、`--trust_remote_code`、`--recipe`、`--qformat`、`--kv_cache_qformat`、`--dataset`、`--calib_size`、`--calib_seq`、`--calib_with_images`、`--export_path`、`--batch_size`、`--device`、`--attn_implementation`、`--use_fsdp2`、`--cpu_offload`、`--low_memory_mode`、`--gpu_max_mem_percentage`、`--skip_generate`。

- [ ] 写 `tests/server/test_capabilities.py`：造一个临时假仓库（`tmp_path/examples/hf_ptq/hf_ptq.py` 里写几行 `parser.add_argument("--qformat", ...)` / `add_argument("--foo_new_flag")`），`extract_flags` 能抽出；`probe_capabilities` 的 `mapped` / `unmapped` / `missing` 分类正确；`validate_cmd_flags("... --qformat nvfp4 --bad_flag", ["--qformat"])` 返回 `["--bad_flag"]`。`git` 命令在测试里 mock 掉（`monkeypatch` `runner.run_shell`），不真跑 git。`.venv/bin/python -m pytest tests/server/test_capabilities.py -q` → 全绿。

- [ ] Commit: `git add server/capabilities.py tests/server/test_capabilities.py && git commit -m "feat(server): capabilities —— 版本 + hf_ptq flag 静态抽取 + 映射表"`

---

## Task 4 — `server/jobs.py`（registry + poller + SSE + 落盘/重挂）

**Files:** `server/jobs.py`、`tests/server/test_jobs.py`、`tests/server/conftest.py`

**Interfaces:**
- Consumes: `webui.runner.{start_task, start_script_task, poll_task, read_log, abort_task, ensure_venv, ensure_eval_env, probe_eval_env}`、`webui.progress.parse` / `parse_install`、`webui.cmds.{build_cmd, build_eval_script, result_tables}`
- Produces（`server/jobs.py`）：
  - `@dataclass class Job`：`id: str`、`kind: str`（`quant` / `eval` / `venv` / `eval_env`）、`task: Optional[runner.Task]`（`venv` / `eval_env` 类没有 `Task`，是 None）、`state: str`（`starting` / `running` / `done` / `failed` / `aborted`）、`log_path: str`（venv 类是 `ensure_venv` 自己 mkstemp 的日志路径——见下方 venv job 说明）、`cmd: str`、`export_dir: str = ""`、`version: str = ""`、`commit: str = ""`、`started_at: float`、`log_offset: int = 0`（SSE 续传游标）、`_subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)`、`_done: bool = field(default=False, repr=False)` / `_exc: Optional[BaseException] = field(default=None, repr=False)`（venv 类包装线程的终态信号——**必须**：`ensure_venv` 在 venv 已存在时早退 `return`，**不写 `VENV_DONE` marker**，只靠 marker 判终态会挂死）
  - `class JobRegistry`：
    - `__init__(self, cfg: AppConfig)`
    - `create(self, kind: str, task: Optional[runner.Task], cmd: str, export_dir: str = "", version: str = "", commit: str = "") -> Job`
    - `get(self, job_id: str) -> Optional[Job]`
    - `list(self) -> list[Job]`（活跃在前，历史在后，按 `started_at` 倒序）
    - `abort(self, job_id: str) -> str`
    - `persist(self, job: Job) -> None`：写 `cfg.local.task_path / job.id / job.json`（`{pid, log_path, cmd, state, export_dir, kind, version, commit, started_at}`；`task is None` 时 `pid` 写 `""`）
    - `reload(self) -> list[Job]`：启动时扫 `cfg.local.task_path/*/job.json`，`pid` 非空且存活（`kill -0`）的重挂回 registry（`state="running"`、起 poller）；pid 空或已死 → 读日志尾判 done/failed（venv 类看 `VENV_DONE rc=` 行，其余看 `done_marker`），不重挂
    - `start_poller(self, job: Job) -> None`：`threading.Thread(target=self._poll, args=(job,), daemon=True, name=f"job-{job.id}")`；`_poll` 的每轮逻辑：**quant/eval** 走 `runner.poll_task(job.task)` + `runner.read_log(job.task, 30000)`，`job.task.state != "running"` 即终态（done/failed）；**venv/eval_env** 不轮询 `Task`，而是 `runner._wait_for_marker` 式地 tail 自己的 `log_path`（marker = `VENV_DONE` / `EVAL_ENV_DONE`），退出条件三选一：(a) marker 行在且 `rc=0` → done；(b) marker 行在且 `rc≠0` → failed；(c) **包装线程已退出**（`job._done=True`）→ `_exc` 非 None 则 failed、否则 done（venv 已存在时 `ensure_venv` 早退不写 marker，靠 (c) 兜底；marker 仅用于日志流与终态佐证）
    - `subscribe(self, job: Job, last_event_id: str = "") -> tuple[asyncio.Queue, int]`：登记订阅者并返回起始 offset（header 值优先，否则 `job.log_offset`）；生成器 `async def sse_stream(self, job, q, start_offset) -> AsyncIterator[str]` 从该 offset 起吐事件：`event: {log|progress|state}` + `id: <offset>` + `data: json`，收到 `state ∈ (done, failed, aborted)` 后 `break`
    - `_broadcast(self, job, evt: str, data)`：`{"type": evt, "data": data, "id": job.log_offset}` 塞进每个订阅者的 `asyncio.Queue`（跨线程用 `loop.call_soon_threadsafe`；订阅者列表在 broadcast 前快照，避免投递中变更）
- Produces（`tests/server/conftest.py`）：`@pytest.fixture def cfg(tmp_path) -> AppConfig`（`task_dir=tmp_path/"tasks"`、`repo=tmp_path/"repo"`、`venv=tmp_path/"venv"`、`hf_cache=tmp_path/"hf"`）、`@pytest.fixture def registry(cfg) -> JobRegistry`

**Steps:**

- [ ] 写 `server/jobs.py` 的 `Job` dataclass + `JobRegistry.create/get/list/abort/persist`。`create` 里 `id = job.kind + "_" + time.strftime("%Y%m%d-%H%M%S") + "_" + uuid4().hex[:6]`（保证唯一；venv 类 job 没有 Task，目录名同 id）。**不写** poller 和 SSE（下一步）。

- [ ] 写 `tests/server/conftest.py`（`cfg` / `registry` fixtures）。

- [ ] 写 `tests/server/test_jobs.py` 的 `create` / `persist` / `reload` 部分：`create` 返回的 job 在 `list()` 里；`persist` 写出 `job.json`（字段齐全）；`reload` 能读回（pid 假的 = 不活，`state` 落到 `done`/`failed`）。用 `monkeypatch` 把 `runner.run_shell` 的 `kill -0` 分支 mock 成「不存活」。`.venv/bin/python -m pytest tests/server/test_jobs.py -q` → 全绿。

- [ ] 写 poller：`start_poller` / `_poll` 按 Task 4 Interfaces 的两分支实现（quant/eval 走 `Task`，venv/eval_env 走 marker-tail + `_done`/`_exc` 退出条件 (a)(b)(c)）。`_poll` 每轮：读日志 → `job.log_offset = len(raw)` → `_broadcast("log", raw)` → `rep = progress.parse(raw)`（venv/eval_env 用 `progress.parse_install(raw.splitlines())`）→ `_broadcast("progress", dataclasses.asdict(rep))` → 结束判定后 `job.state = ...`、`self.persist(job)`、`_broadcast("state", job.state)` 退出循环；循环中 `time.sleep(5)`。

- [ ] 写 `subscribe` + `sse_stream`：SSE 行格式固定为 `event: <type>\nid: <offset>\ndata: <json>\n\n`；`data` 的 json 里 log 事件是 `{"tail": raw[-30000:]}`、progress 事件是 `ProgressReport` 的 `asdict`（`bar` 字段丢弃，前端自己画）、state 事件是 `{"state": job.state}`。

- [ ] 写 `tests/server/test_jobs.py` 的 poller + SSE 部分：用 `unittest.mock.patch` 把 `runner.poll_task` / `runner.read_log` 改成受控行为（先 `running` + 假日志，后 `done`），`asyncio.run` 跑 `sse_stream` 收集到 `log` / `progress` / `state` 三种事件，`id` 单调递增；另测一个 venv 类 job（`task=None`，`log_path` 指向临时文件，先写 `installing...` 再写 `VENV_DONE rc=0`）能走到 `done`；再测早退路径：venv 目录已存在（`ensure_venv` 提前 return 不写 marker），包装线程置 `_done=True`（无 `_exc`），poller 也能走到 `done`。`.venv/bin/python -m pytest tests/server/test_jobs.py -q` → 全绿。

- [ ] Commit: `git add server/jobs.py tests/server && git commit -m "feat(server): jobs registry + poller + SSE + 落盘/重挂"`

---

## Task 5 — `server/api.py` 只读端点（bootstrap / probe / vocab / capabilities / preview / jobs / jobs/{id} / jobs/{id}/events / eval/results / paths / net / models/local / artifacts）

**Files:** `server/api.py`（扩）、`tests/server/test_api_smoke.py`（扩）

**Interfaces:**
- Consumes: `webui.runner.{probe_host, probe_eval_env, fetch_eval_results, read_log}`、`webui.vocab.{scan_qformats, scan_datasets, all_qformats, kv_choices, DATASETS, COMBOS}`、`webui.cmds.{build_params, build_cmd, result_tables, EVAL_TASKS, EVAL_PRESETS}`、`server.capabilities.{probe_capabilities, validate_cmd_flags}`、`server.jobs.JobRegistry`、`server.schemas.*`
- Produces（`server/api.py` 的 `APIRouter`，本步实现的端点）：
  - `GET /api/bootstrap` → `{paths: {repo, venv, task_dir, hf_cache}, net: {mode, proxy}, probe: HostStatus, vocab: {qformats, kv, datasets, combos}, capabilities: dict, jobs: [Job 摘要], defaults: dict}`
  - `POST /api/probe` → `{probe: HostStatus, vocab: {qformats, kv}}`（重扫词表）
  - `GET /api/vocab` → `{qformats: [...], kv: [[label, value], ...], datasets: [{key, desc}], combos: {name: desc}}`
  - `GET /api/capabilities` → `probe_capabilities(cfg)` 的 dict
  - `POST /api/preview`（body `QuantizeParams`）→ `{cmd: str, pretty: str, missing_flags: list[str]}`；`missing_flags` 非空时 HTTP 400 + `error`
  - `GET /api/jobs` → `{active: [Job 摘要], history: [Job 摘要]}`
  - `GET /api/jobs/{id}` → `{job: 摘要, state, log_tail: str, progress: ProgressReport, result_tables: list[list[str]] | null}`
  - `GET /api/jobs/{id}/events` → **SSE**（`media_type="text/event-stream"`，`StreamingResponse`）；请求头 `Last-Event-ID` 续传；`GET` 但流式长连接
  - `GET /api/eval/results` → `{rows: list[list[str]], note: str, files: list[str]}`（`result_tables(results)` 一次性消费**全部** results——`results[0]`=量化、`[1]`=基线——只返回**单张**对照表 + note，不要把同一张表塞进每个 file）
  - `GET /api/paths` → `{paths: {repo, venv, task_dir, hf_cache}}`（读 `CFG.local` 的当前值，不解析绝对路径——原样回 UI 已填的字符串）
  - `GET /api/net` → `{net: {mode, proxy}}`（读 `CFG.net`）
  - `GET /api/models/local` → **C2 最小版**：扫 `CFG.local.hf_cache_path / "hub" / models--*--*`，返 `{models: [{id: "org/name", size: "5.7G", stub: bool}]}`；`stub` = 目录只有 `refs`（几 KB）的如实标注；扫失败返 `{models: [], error: str}`
  - `GET /api/artifacts` → **C1 最小版**：扫 `CFG.local.task_path/*` 子目录，每个有 `run.log` 的算一条，返 `{artifacts: [{dir, model, ts, size: str, state: str}]}`；`state` 从同目录 `job.json` 读（无则 `"manual"`）；`size` 用 `runner.run_shell(f"du -sh {q_remote(dir)}")`
  - `Job 摘要` = `{id, kind, state, started_at, export_dir, version, commit, cmd_preview: str[:120]}`

**Steps:**

- [ ] 扩 `server/api.py`：`router = APIRouter(prefix="/api")`；模块级 `CFG = load_config()`、`REGISTRY = JobRegistry(CFG)`（`REGISTRY.reload()` 一次）。每个端点 `@router.get/post`。

- [ ] 实现 `GET /api/bootstrap`：`probe = runner.probe_host(CFG)`；`vocab` 用 `runner` 的 `scan_qformats(CFG.local.repo)`（异常时回落内置）+ `vocab.kv_choices()` + `vocab.DATASETS` + `vocab.COMBO_DESC`；`capabilities = probe_capabilities(CFG)`；`jobs = REGISTRY.list()`。

- [ ] 实现 `POST /api/probe`：`runner.probe_host(CFG)` + 重扫词表（同 bootstrap 的 vocab 逻辑，抽成 `_scan_vocab() -> dict` 复用）。

- [ ] 实现 `GET /api/vocab` / `GET /api/capabilities`：直接调 `vocab.*` / `probe_capabilities(CFG)`。

- [ ] 实现 `POST /api/preview`：`p = req.to_build_params_dict()`；`cmd = build_cmd(p, CFG)`；`pretty = _prettify_cmd(cmd)`（把 `app.py:1020` 的 `_prettify_cmd` 搬进 `cmds.py`，`app.py` 也改成 import 它）；`flags_all = probe_capabilities(CFG)["flags_all"]`；`missing = validate_cmd_flags(cmd, flags_all)`；`if missing: raise HTTPException(400, detail={"code": "unknown_flags", "message": "本版本没有这些参数", "detail": missing})`；否则 `return {"cmd": cmd, "pretty": pretty, "missing_flags": []}`。

- [ ] 实现 `GET /api/jobs` / `GET /api/jobs/{id}`：`REGISTRY.list()` / `REGISTRY.get(id)`（404 若无）；`log_tail = runner.read_log(job.task, 30000)`（venv 类 `task is None` 时 `tail_log(job.log_path)`）；`progress = progress.parse(log_tail)`（venv 类用 `parse_install`）；`result_tables = cmds.result_tables(runner.fetch_eval_results(str(CFG.local.task_path))) if job.kind == "eval" else None`。

- [ ] 实现 `GET /api/jobs/{id}/events`：`job = REGISTRY.get(id)`（404）；`q, start = REGISTRY.subscribe(job, last_event_id=request.headers.get("Last-Event-ID", ""))`；`return StreamingResponse(REGISTRY.sse_stream(job, q, start), media_type="text/event-stream")`。`@app.on_event("shutdown")` 时清理订阅者。

- [ ] 实现 `GET /api/paths` / `GET /api/net`：直接读 `CFG.local` / `CFG.net` 的原始字符串字段。

- [ ] 实现 `GET /api/models/local`：`hub = CFG.local.hf_cache_path / "hub"`；`if not hub.exists(): return {"models": [], "error": f"缓存目录不存在: {hub}"}`；遍历 `models--*--*` 子目录，`id = name.replace("models--", "").replace("--", "/", 1)`；`size = runner.run_shell(f"du -sh {q_remote(d)}")[1].split()[0]`；`stub = (not (d / "snapshots").exists())`（没有 snapshots 子目录 = 只有 refs 的空壳）。

- [ ] 实现 `GET /api/artifacts`：`base = CFG.local.task_path`；`if not base.exists(): return {"artifacts": []}`；遍历子目录，有 `run.log` 的算一条；`job.json` 存在则读 `state` / `model` / `started_at`，否则 `state="manual"`。

- [ ] 实现 `GET /api/eval/results`：`results = runner.fetch_eval_results(str(CFG.local.task_path))`；`rows, note = result_tables(results)`（单表消费全部 results，见 Interfaces）；`return {"rows": rows, "note": note, "files": [r["file"] for r in results]}`。

- [ ] 写 `tests/server/test_api_smoke.py`：用 `TestClient`（`fastapi.testclient`，`app` 从 `server.api` import），`monkeypatch` 掉 `runner.probe_host` / `probe_capabilities` / `runner.read_log` / `runner.fetch_eval_results` / `vocab.scan_qformats` / `runner.run_shell`（返回固定值），断言全部只读端点的 JSON 形状（字段名齐全、HTTP 200）：bootstrap / probe / vocab / capabilities / jobs / jobs/{id} / eval/results / paths / net / models/local / artifacts。`POST /api/preview` 的 `missing_flags` 非空时 400。`GET /api/jobs/{id}/events` 用 `client.stream` 断言首行是 `event:` 且 `content-type` 含 `text/event-stream`。`.venv/bin/python -m pytest tests/server/test_api_smoke.py -q` → 全绿。

- [ ] 验证：`.venv/bin/python serve.py &` → `curl -s http://127.0.0.1:7862/api/bootstrap` / `/api/vocab` / `/api/capabilities` / `/api/paths` / `/api/net` / `POST /api/probe` / `POST /api/preview`（body `{"model":"x","qformat":"nvfp4",...}`）→ 全 200 + 字段齐全；`kill %1`。

- [ ] Commit: `git add server/api.py tests/server/test_api_smoke.py && git commit -m "feat(server): 只读端点 bootstrap/probe/vocab/capabilities/preview/jobs/events/paths/net/models/artifacts"`

---

## Task 6 — `server/api.py` 写端点（quantize / eval / venv / eval-env / jobs/{id}/abort / recipe/diff / eval/baseline / paths / net）

**Files:** `server/api.py`（扩）、`tests/server/test_api_smoke.py`（扩）

**Interfaces:**
- Produces（本步新增端点）：
  - `POST /api/quantize`（body `QuantizeParams`）→ `202 {job_id, job: 摘要}`；**前置校验**：`probe_host(CFG).reachable`（否则 400 `unreachable`）+ `venv_ok`（否则 400 `venv_missing`）+ `is_local_model` 时 `check_local_model`（否则 400 `bad_model_path`）+ `gpu_busy` 时**不拒绝**，返回 `202` 但带 `warn_gpu: true`（ADR 023 不抢占，靠前端二次确认）
  - `POST /api/eval`（body `EvalStart`）→ `202 {job_id, job}`；前置 `probe_eval_env(CFG)`（失败 400 `eval_env_missing`）
  - `POST /api/venv` → `202 {job_id, job}`（后台 `ensure_venv`，`kind="venv"`）
  - `POST /api/eval-env` → `202 {job_id, job}`（后台 `ensure_eval_env`，`kind="eval_env"`）
  - `POST /api/jobs/{id}/abort` → `{message: str}`（404 若无）
  - `POST /api/recipe/diff`（body `RecipeDiffRequest`）→ `{summary: QuantSummary, groups: list, diff: str, files: str}`
  - `POST /api/eval/baseline`（body `{quant_path: str}`）→ `{baseline: str, message: str}`
  - `PUT /api/paths`（body `PathsRequest`）→ `{message: str}`（写回 `config.yaml`）
  - `PUT /api/net`（body `NetRequest`）→ `{message: str}`（写 `net.json`）
  - `POST /api/net/test` → `{ok: bool, detail: str}`

**Steps:**

- [ ] 实现 `POST /api/quantize`：`p = req.to_build_params_dict()`；`cmd = build_cmd(p, CFG)`（`ValueError` → 400）；前置校验（reachable / venv_ok / check_local_model）；`version, commit = probe_capabilities(CFG)["version"], [...]["commit"]`；`task = runner.start_task(CFG, CFG.local.task_dir, task_slug(model), export_path, cmd, runner.proxy_for(CFG), lambda s: None)`；`job = REGISTRY.create("quant", task, cmd, export_dir=export_path, version=version, commit=commit)`；`REGISTRY.persist(job)`；`REGISTRY.start_poller(job)`；`return 202 {job_id: job.id, job: 摘要, warn_gpu: probe.gpu_busy}`。

- [ ] 实现 `POST /api/eval`：前置 `probe_eval_env`；`script = build_eval_script(quant, baseline, tasks, tp, port, batch, CFG)`；`tdir = CFG.local.task_path / f"eval_{time.strftime('%Y%m%d-%H%M%S')}"`；`task = runner.start_script_task(CFG, str(tdir), script, runner.proxy_for(CFG), lambda s: None, pkill_pattern="vllm serve")`；`job = REGISTRY.create("eval", task, "bash run.sh", version=version, commit=commit)`；persist + `REGISTRY.start_poller(job)`；`return 202`。

- [ ] 实现 `POST /api/venv` / `POST /api/eval-env`：**不改** `runner.ensure_venv`/`ensure_eval_env` 的签名（约束：不碰 runner）——做法是包装：先自己 `tempfile.mkstemp(prefix="modelopt_api_venv_")` 拿到 `log_path` 写进 `job.log_path`，`task=None`，`cmd` 写 `"ensure_venv"`/`"ensure_eval_env"` 标识；再 `job = REGISTRY.create("venv", None, "ensure_venv")`、`job.log_path = <mkstemp 路径>`、persist、`start_poller(job)`；最后开一个一次性包装线程跑 `runner.ensure_venv(lambda s: None, CFG)`，**正常退出置 `job._done = True`，异常时先 `job._exc = e` 再 `_done = True`**（poller 的退出条件 (c) 靠这两个字段判终态，见 Task 4）。**但注意** `ensure_venv` 内部会自己另 mkstemp 写日志，外层线程跑完后 `job.json` 的 log_path 不是真实日志 ⇒ 正确做法：poller 的 venv 分支不依赖 job.log_path，而是每轮 `glob /tmp/modelopt_venv_setup_*.log` 取 mtime 最新且晚于 job.started_at 的那个（`ensure_venv` 的 prefix 固定，本机单用户足够可靠）；`job.log_path` 在 poller 首轮定位后回填并 persist。`ensure_eval_env` 同（prefix `modelopt_eval_env_`）。`return 202`。

- [ ] 实现 `POST /api/jobs/{id}/abort`：`REGISTRY.abort(id)`（`task is not None` 时 `runner.abort_task(job.task)`；`task is None`（venv 类）时只把 `state` 置 `aborted` + 置 `job._abort_flag`，poller 下轮见到 flag 就退出、不写 done）+ persist；`return {"message": "已发送中止信号"}`（404 若无）。

- [ ] 实现 `POST /api/recipe/diff`（对齐 legacy `do_validate` 的输入形状）：`job = REGISTRY.get(req.job_id)`（404）；`log_tail = runner.read_log(job.task, 5000)`；`hint = ""`，`contextlib.suppress` 里 `hint = job.task.remote_cmd.split("--export_path")[-1].split()[0].strip("'\"")`；`mine = validate.scan_export_dir(log_tail, export_dir_hint=hint)`；`if req.official: off = validate.fetch_official_summary(req.official, proxy=runner.proxy_for(CFG)); diff = "\n\n".join(validate.diff_summaries(mine, off))`；`return {"summary": dataclasses.asdict(mine), "groups": mine.groups, "diff": diff, "files": mine.files}`。

- [ ] 实现 `POST /api/eval/baseline`：读 `quant_path/config.json`（`runner.run_shell(f"cat {q_remote(quant_path)}/config.json")`），抽 `_name_or_path`（同 `app.py:1003` 的 `do_fetch_baseline` 逻辑）；`return {"baseline": b, "message": "..."}`。

- [ ] 实现 `PUT /api/paths` / `PUT /api/net` / `POST /api/net/test`：`config.save_local` / `save_net` / `runner.test_proxy(CFG)`（同 `app.py` 的 `do_save_env` / `do_save_net` / `do_test_net`）。

- [ ] 扩 `tests/server/test_api_smoke.py`：`POST /api/quantize` 的 `venv_ok=False` 时 400；`gpu_busy=True` 时 202 + `warn_gpu`；`POST /api/eval` 的 `eval_env_missing` 400；`POST /api/jobs/{id}/abort` 404 且对 venv 类 job（task=None）返 200；`POST /api/recipe/diff` 对不存在的 job_id 404；`PUT /api/paths` / `PUT /api/net` 写文件（`tmp_path`）；`POST /api/net/test` 200。`monkeypatch` 掉 `runner.start_task` / `start_script_task` / `probe_host` / `probe_eval_env` / `test_proxy` / `read_log`。`.venv/bin/python -m pytest tests/server/test_api_smoke.py -q` → 全绿。

- [ ] 验证：`.venv/bin/python serve.py &` → `POST /api/quantize`（假模型）→ 400 `bad_model_path` 或 `venv_missing`（mac 无 venv）；`PUT /api/net` → 200 + `net.json` 写出；`kill %1`。

- [ ] Commit: `git add server/api.py tests/server/test_api_smoke.py && git commit -m "feat(server): 写端点 quantize/eval/venv/eval-env/abort/recipe/paths/net"`

---

## Task 7 — `server/static.py` + token 中间件 + 全量回归 + GPU 机验证

**Files:** `server/static.py`、`server/api.py`（扩，挂静态 + 中间件）、`requirements.txt`（确认）、`docs/superpowers/plans/2026-10-14-plan2-fastapi-server.md`（本文件，写验证记录）

**Interfaces:**
- Produces（`server/static.py`）：`def mount_static(app: FastAPI) -> None`：`web/dist` 存在时 `app.mount("/", StaticFiles(directory="web/dist", html=True))`（SPA fallback：非 `/api/*` 且无对应文件时返 `index.html`）；不存在时 `app.get("/")` 返回 `{"error": {"code": "no_dist", "message": "web/dist 不存在，先 npm run build", "detail": "GET /api/* 仍可用"}}`（**不白屏**）。
- Produces（`server/api.py` 扩）：token 中间件 `@app.middleware("http")`，`CFG.server_token` 非空时校验 `X-Api-Token`（仅 `/api/*`，静态文件不校验），不符返 401 `{error: {code: "bad_token", ...}}`。

**Steps:**

- [ ] 写 `server/static.py`：`mount_static(app)`（`web/dist` 路径用 `Path(__file__).parent.parent / "web" / "dist"`）。`server/api.py` 里 `app = FastAPI(...)` 后调 `mount_static(app)`。

- [ ] 写 token 中间件：`@app.middleware("http") async def _token(req, call_next): if CFG.server_token and req.url.path.startswith("/api") and req.headers.get("X-Api-Token") != CFG.server_token: return JSONResponse(401, {"error": {...}}); return await call_next(req)`。

- [ ] 全量回归：`.venv/bin/python -m pytest -q` → **所有** golden + server 测试全绿。`.venv/bin/python -c "import app; app._build_app()"` → `BUILD OK`。`.venv/bin/python -c "import server.api; print('API OK')"` → `API OK`。

- [ ] **GPU 机验证**（`chrisqian@192.168.124.39`）：
  - `./deploy.sh`（rsync 到 GPU 机，`--exclude='.venvs/'`）
  - GPU 机 `uv pip install --python .venv/bin/python fastapi uvicorn "pydantic>=2" httpx`（装进 UI 用 `.venv`）
  - GPU 机 `nohup .venv/bin/python serve.py >/tmp/moweb_api.log 2>&1 </dev/null &`
  - `curl http://127.0.0.1:7862/api/bootstrap` → 200 + `probe.gpu` = RTX 5090 D + `probe.venv_ok` = true（Plan 1 已建）
  - `curl -X POST http://127.0.0.1:7862/api/probe` → 200
  - `curl http://127.0.0.1:7862/api/capabilities` → 200 + `version` = 真实 ModelOpt 版本 + `flags_all` 35 项 + `mapped` 18 项 + `unmapped` 17 项
  - `curl -X POST http://127.0.0.1:7862/api/preview -d '{"model":"Qwen/Qwen3-8B","qformat":"nvfp4","datasets":["cnn_dailymail"],"dataset_sizes":{"cnn_dailymail":1024},"calib_seq":512,"calib_with_images":false,"export_path":"","batch_size":1,"device":"cuda","attn_implementation":"sdpa","use_fsdp2":false,"cpu_offload":false,"low_memory_mode":false,"gpu_max_mem_percentage":0.0,"skip_generate":false,"trust_remote_code":true,"recipe":"","kv_cache_qformat":"none","custom_datasets":[]}' -H 'Content-Type: application/json'` → 200 + `cmd` 与 golden 逐字节对齐（路径不同是预期，`Qwen3-8B` vs golden 的 `Llama-3.1-8B`）
  - 确认 legacy 7861 仍跑（`curl http://127.0.0.1:7861/` → 200）+ 新 7862 双开
- [ ] 浏览器（mac 上）逐项确认 `/api/bootstrap` / `/api/capabilities` / `/api/preview` 的 JSON 形状与设计文档 §5.3/§7 一致（**未做**：mac 无 agent-browser，后端直验替代，如实记录）

- [ ] 本文件末尾写「GPU 机验证记录」（日期、版本、各端点实测结果、遗留）。

- [ ] Commit: `git add server/static.py server/api.py requirements.txt && git commit -m "feat(server): static 托管 + token 中间件 + GPU 机验证记录"`

---

## Task 8 — 收尾

- [ ] 残留 grep：`grep -rn "import gradio\|gr\." server/ serve.py` → 期望 **0**（server 层零 Gradio 依赖）
- [ ] `grep -rn "webui.cmds" app.py serve.py server/` → 确认三处都从 `cmds.py` import（没有谁还在 import `app.build_cmd`）
- [ ] `git log --oneline` 确认提交序列连贯
- [ ] 记忆同步：`03-Decisions.md` ADR 024 状态行追加「Plan 2（FastAPI + jobs/SSE + capabilities）已实现并验证（2026-10-14，N commits）」；`05-Active-Context.md` ModelOpt 行就地改写（≤6144B）
- [ ] Commit: `git commit -m "docs: Plan 2 收尾 —— 残留终审 + 记忆同步"`

---

## 自检（写计划时已做）

- **spec 覆盖**：§5.3 的 22 端点全在（bootstrap/probe/vocab/capabilities/preview/quantize/eval/venv/eval-env/jobs/jobs/{id}/jobs/{id}/abort/jobs/{id}/events/recipe/diff/eval/baseline/eval/results/models/local/artifacts/paths GET+PUT/net GET+PUT+net/test）—— **`models/local` 与 `artifacts` 本计划只做最小版**（`GET /api/models/local` 扫 `hf_cache/hub` 返 `[{id, size, stub}]`；`GET /api/artifacts` 扫 `task_dir/*/` 返 `[{dir, model, ts, size}]`），Plan 4 扩成 C1/C2 完整功能。
- **placeholder 扫描**：无 TBD/TODO/"Similar to Task N"；每个代码步骤给完整签名与关键代码块。
- **类型一致性**：`build_params(a, cfg)` / `build_cmd(p, cfg)` / `build_eval_script(..., cfg)` 的 `cfg` 参数在 Task 1 引入，Task 5/6 的调用点全部带上；`Job` dataclass 的 `task: Optional[runner.Task]` 覆盖 `venv` / `eval_env`（无 `Task`）的情况。

---

## 执行选项

1. **subagent-driven**（推荐）：每个 Task 派一个 `acp_delegate` worker，按 Task 0→8 顺序，每完成一个 Task 跑该 Task 的验证步骤 + commit，下一个 Task 基于前一个的 commit。Task 7 的 GPU 机验证要 worker 能 ssh（key auth 已配好）。
2. **inline**：主会话逐 Task 执行，每 Task 的验证步骤跑完再进下一个。适合想边看边调的情况。

---

## GPU 机验证记录（2026-10-14 19:07）

机器：`chrisqian@192.168.124.39`（chrisqian-ubuntu01，RTX 5090 D 32GB）。部署：`./deploy.sh`（rsync，`--exclude='.venvs/'`）。GPU 机 UI 用 `.venv` 走 `~/.local/bin/uv`（`python3 -m venv` 不可用，无 ensurepip）：`uv venv .venv` + `uv pip install --python .venv/bin/python fastapi uvicorn "pydantic>=2" httpx`（代理 `http://192.168.124.24:7890` 三变量）。

端点实测（7862 serve.py，legacy 7861 双开正常）：

| 端点 | 结果 |
|---|---|
| `GET /api/bootstrap` | 200；`probe.gpu`=RTX 5090 D、`venv_ok`/`repo_ok`/`reachable` true、`hf_direct_ok` false、`disk_free` 437G、`hf_cache` 5.7G；keys=capabilities/defaults/jobs/net/paths/probe/vocab |
| `POST /api/probe` | 200 |
| `GET /api/capabilities` | 200；`version`=0.48.0rc0-35-g254195350、`commit`=254195350、`flags_all`=**34**、`mapped`=18、`unmapped`=**16**、`missing`=0、`error`="" |
| `POST /api/preview` | 200；4/4 golden 用例去路径前缀后 flag 串**逐字节相等**（路径差异 = 本机 CFG，预期） |
| `GET /`（no_dist） | 200 `{error:{code:no_dist}}`，API 照常 |
| legacy `GET :7861/` | 200（双开对照未破坏） |

**验证中发现并修复（2 个真 bug，单测全 mock 未暴露，靠实机抓出）：**

1. **`~` 未展开**：`server/capabilities.py` 用裸 `Path(cfg.local.repo)`，git 走 `q_remote` 所以 version 正常，但 `extract_flags`/`exists()` 用 `~/Model-Optimizer` 字面路径 ⇒ `flags_all=0`、`error="hf_ptq.py 不存在: ~/..."`。修：`repo = cfg.local.repo_path`（`resolve_relative` 含 expanduser）。回归测试 `test_probe_capabilities_tilde_repo`。与 Plan 1 BUG 3（`scan_qformats` 同病）同类。
2. **`custom_datasets` 条数 int 被拒**：schema `list[list[str]]` 拒绝 `[['mysite', 128]]`（gradio 表格/前端会发 int）⇒ preview 422。修：`list[list[str | int]]`（`cmds.py:156` 用 `str(row[0])` 取名称，int 安全）。回归测试 `test_quantize_params_custom_datasets_mixed_types`。

**与计划正文的偏差（1 处，以实测为准）：** 本 Task 7 步骤写「`flags_all` 35 项 + `unmapped` 17 项」（沿用设计稿/ADR 024 的手工清单），实测静态抽取 = **34 / 16**（本地脚本独立对 GPU 机 hf_ptq.py 抽取双向 diff 为空，与 API 完全一致）⇒ 手工清单多算 1 个，真实以 34/16 为准。16 个 unmapped：--specdec_offline_dataset --inference_tensor_parallel --inference_pipeline_parallel --awq_block_size --sparsity_fmt --export_fmt --use_seq_device_map --verbose --auto_quantize_checkpoint --kv_auto_quantize_checkpoint --moe_calib_experts_ratio --vllm_fakequant_export --cast_mxfp4_to_nvfp4 --offload_folder --max_cpu_memory_gb --max_gpu_memory_gb。

**未做（如实记录）：** mac 无 agent-browser（missing-binary），浏览器逐项确认改用 TestClient + curl 直验替代；token 中间件未上 GPU 机实测（config.yaml `server.token` 默认空，单测覆盖）。

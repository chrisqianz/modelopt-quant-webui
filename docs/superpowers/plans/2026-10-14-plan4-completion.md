# Plan 4 · 收尾 —— 逃生口 / C1C2 完整版 / 显存估算器 / 单入口删 legacy

**日期**：2026-10-14
**基线**：`8a2ebdc`（Plan 3 收尾，91 passed，GPU 机 7861 legacy + 7862 新 UI 双开）
**依据**：设计稿 `docs/superpowers/specs/2026-10-14-fastapi-react-ui-design.md` §7 层 3 / §8（C1/C2）/ §6 估算器 / §10 第 6-7 步 / D2 / D8。m00225 批准的第四份计划。

## 范围与硬约束

- **不真跑量化/评测**（ADR 023：GPU 被 vllm serve 占卡，不抢占）；真跑验证只记「待显存空闲」。
- golden 四用例**逐字节不变**（逃生口只加不减：新增用例走 `--update-golden`）。
- `webui/progress.py`、`vocab.py`、`validate.py` 不动；`webui/cmds.py` 只许加 extra_args 拼接。
- 全程 **inline 执行**（m03282 用户指令：把控门控时间，否则 inline 自己做）。
- 单入口按 D2：`python app.py` 起 uvicorn 7861，同端口挂 `/api/*` + SSE + 静态 `web/dist`；legacy Gradio 版先改名 `legacy_app_gradio.py` 留档（D8），删文件放在**最后一步**，中间保持双开对照。

## 现状事实（写计划时核过，执行前不必重查）

- `extra_args` 目前**只存不拼**：`server/schemas.py:45` 注释「本计划只存不拼」，`webui/cmds.py build_cmd` 不消费它；`QuantizeView.tsx:366` 已有输入框但填了没用。
- `capabilities.unmapped` 已有 16 个 flag（GPU 机实测 34 总 / 18 mapped / 16 unmapped），`bootstrap` 已吐给前端，但**未渲染成可点标签**。
- C1 `GET /api/artifacts` 最小版：`{dir, model, ts, size, state}`；`job.json` 里**有 `cmd` 字段**可解析 `--qformat`；`model` 字段存的其实是 `export_dir`（写端点塞进去的）。
- C2 `GET /api/models/local` 最小版：`{id, size, stub}`（stub = 无 snapshots 目录）；**无 snapshot revision**（`hub/models--*--*/refs/main` 里有一行 revision）；`QuantizeView` datalist 自动补全已有，但**无「点击填入」按钮**、stub 未标注。
- 估算器：设计稿 §6 给了真实数据方案（权重量 du / HF API；NVFP4 固定折算；KV 按 config.json 层数/头维度，取不到退经验公式 `modelSize*0.15*(ctx/2048)*batch` 并标注「估算」；输出对真实显存总量给「放得下 / 还差 X GB」）。`TopBar.tsx` 目前无估算器按钮。
- 入口：`serve.py` = `CFG.server_port + 1`（7862）；legacy `app.py` 732 行 Gradio；`tests/golden_adapter.py` 依赖 `app.INPUT_ORDER` + `app.CFG`（删 legacy 前必须先把这两处搬到独立模块）。
- `requirements.txt` 含 `gradio>=4.44`（删 legacy 后移除）；GPU 机 venv 走 uv（`python3-venv` 未装）。
- `AGENTS.md` 不存在（无此文件可更新）；README「目录/快速开始」双开描述需改。

## Task 1 · 附加参数逃生口接线（§7 层 3，最关键）

**Files:** `webui/cmds.py`、`server/api.py`、`tests/`、`web/src/views/QuantizeView.tsx`

- `build_cmd(p, cfg)` 尾部：`extra = (p.get("extra_args") or "").strip()`，非空时 append 到 parts。**校验先行**：
  - 禁字符集 `;&|<>\n$\``（命令操作符全禁，防注入；`shlex.split` 后再逐 token 拒 `--` 开头之外的裸命令）→ 违规 `raise ValueError("附加参数含禁用字符: ...")`。
  - 每个 token 必须 `--flag` 或 `--flag=value` 或紧跟前一个 flag 的值（即 `shlex.split` 的偶数位是 flag）——简化为：**token 首字符不是 `-` 时必须是前一个 token 的值**，实现上直接 `shlex.split` 成功 + 禁字符即可，不做偶数位校验（flag 值里带 `/` `:` 都合法，过度校验反伤可用性）。
- `server/api.py`：`build_cmd` 的 `ValueError` 已在 quantize/preview 两处被 catch 成 400 `bad_params` ⇒ 校验失败自动 400，**无需新端点**。`unknown_flags` 校验在拼接**后**跑 ⇒ 逃生口填 unmapped flag（在 `flags_all` 里）不会被误杀，填真不存在的 flag 会 400 列出——「上游新功能当天可用」闭环成立。
- golden：`tests/cases.py` 加 1 用例 `extra_args_unmapped`（`extra_args="--verbose --max_gpu_memory_gb 24"` 之类），`--update-golden` 生成并人眼过；原 4 用例 golden **不动**。
- 前端：`QuantizeView.tsx` 在高级区把 `data.capabilities.unmapped` 渲染成可点标签（mono 小徽章，点击 `set('extra_args', 追加 " " + flag)`，已存在则不重复），标签区标题「本版本未映射的参数（点击加入附加参数）」；caps 缺失/全映射时不渲染。
- 验收：pytest 全绿（含新 golden 用例）+ GPU 机 preview 带 `extra_args` 的 4 用例逐字节对齐 + 非法 `extra_args="a;b"` 返 400 bad_params。

## Task 2 · C1 产物库完整版（§8）

**Files:** `server/api.py`、`web/src/api/types.ts`、`web/src/views/ArtifactsView.tsx`

- 后端 `GET /api/artifacts` 每卡加两字段：
  - `qformat`：从 `job.json.cmd`（manual 的从 `run.log` 首行）正则 `--qformat (\S+)` 提取，取不到 `""`；`model` 字段改存**模型名**（从 cmd `--pyt_ckpt_path (\S+)` 提取），`export_dir` 原值留在 `dir`。
  - 大小保留 `du`。
- 前端：卡片显 qformat 徽章 + 模型名 + 时间 + 大小 + 路径；新增「生成 vLLM 启动命令」按钮 → 纯前端模板拼串（设计稿 L944 也是前端行为，不占接口）：
  ```bash
  vllm serve <abs_export_dir> --served-model-name <slug> --dtype auto --gpu-memory-utilization 0.9
  ```
  `abs_export_dir` = `task_dir/dir` 展开（bootstrap.paths 给原始串，前端只做 `~` → `$HOME` 的展示展开，模板用绝对路径占位注释）；拼进 `LogPanel` 同款可复制 `<pre>` + 复制按钮。
- 验收：GPU 机 artifacts 端点返回新字段（当前 1 个 manual 目录或空态都验）+ tsc app 0 错 + build 通。

## Task 3 · C2 本机缓存模型列表完整版（§8）

**Files:** `server/api.py`、`web/src/api/types.ts`、`web/src/views/QuantizeView.tsx`

- 后端 `models_local` 每条目加 `revision`：读 `d/refs/main` 一行（不存在 `""`）；`stub` 判定不变（无 snapshots）。
- 前端模型输入区：datalist 保留（自动补全），**加候选下拉列表**（`modelsLocal` 结果渲染成按钮列表，前 8 条 + 「全部 N 个」），每条点击 → `set('model', id)` 填入输入框；stub 条目渲染成灰显 +「空壳」角标，点击照样可填（用户可能就是要重新下载）。
- 验收：GPU 机 `/api/models/local` 带 revision（本机有 5.7G 缓存可验）+ tsc/build 通。

## Task 4 · 显存估算器（设计稿 §6 修订版）

**Files:** `server/estimate.py`（新）、`server/api.py`、`web/src/components/VramPanel.tsx`（新）、`web/src/components/TopBar.tsx`、`web/src/App.tsx`、`tests/server/test_estimate.py`（新）

- 后端 `GET /api/estimate?model=&qformat=&kv_qformat=&ctx=&batch=`：
  1. 权重量：本机缓存命中（`hub/models--org--name`）→ `du -sb` 缓存目录；未缓存 → HF API `https://huggingface.co/api/models/<id>` 的 `size` 字段（走 `proxy_url_for(CFG)` 代理，超时 5s）；两者都失败 → 返回 `weight_bytes=0, source="unknown"`。
  2. 量化后权重折算系数（文件头常量表，可改）：`nvfp4*/fp4* ≈ 0.55`（4bit + scale）、`fp8* ≈ 1.0`、`int4* ≈ 0.45`、其他 1.0。`quant_bytes = weight_bytes * factor`。
  3. KV：读缓存内 `snapshots/<rev>/config.json` 的 `num_hidden_layers` / `num_key_value_heads` / `head_dim`（缺 `head_dim` 用 `hidden_size // num_attention_heads`）→ `kv_bytes = 2 * layers * kv_heads * head_dim * ctx * batch * kv_elem_bytes`（fp8* 1B、nvfp4* 0.5B、none 0）。config.json 取不到 → `kv_bytes = 0` 且 `source` 追加 `kv_estimate`（退回经验公式项 `weight_bytes * 0.15 * ctx/2048 * batch` 计入并标注「估算」）。
  4. 输出 `{weight_bytes, source, quant_bytes, kv_bytes, vram_total_mib, vram_used_mib, need_gib, fits, short_gib, note}`；`vram_*` 来自 `runner.probe_host(CFG)` 的 `gpu_mem`（「30915 MiB / 32607 MiB」解析）；`fits = total - used - need > 0`。
- 前端：`TopBar` 加「显存估算」按钮 → 右侧滑面板（`VramPanel`，fixed 右 0，宽度 360，黑底白字风格对齐模版开场）：输入 ctx/batch/kv_qformat（model/qformat 从 QuantizeView 当前状态传入，无则面板内自填）→ 调 `/api/estimate` → 大字「放得下 ✓ / 还差 X GB ✗」+ 分项（权重/量化后/KV/余量）+ source 标注（cache/hf_api/估算）+ 真实显存总量与当前占用条。
- 验收：单测 mock du/HF API 全路径；GPU 机对已缓存模型真实估算（权重走 du、KV 走 config.json）+ 未缓存模型走代理 HF API（或如实记 unknown）。

## Task 5 · 单入口迁移（D2：7861 化）

**Files:** `app.py` → `legacy_app_gradio.py`（git mv）、`serve.py` → `app.py`（内容改）、`tests/golden_adapter.py`、`tests/conftest.py`（如引用 app）、`vite.config.ts`、`deploy.sh`、README

- **先拔 golden 对 app.py 的依赖**：`tests/golden_adapter.py` 的 `app.INPUT_ORDER` / `app.CFG` 两处 → 新建 `webui/input_order.py`（`INPUT_ORDER` 20 项照抄现 app.py:53 + `build_initial_config()` 工厂），adapter 改 import 它；跑 pytest 确认 4 用例逐字节不变再动 app.py。
- `app.py` 改名 `legacy_app_gradio.py`（git mv，内容不动）；`serve.py` 内容改为 7861（删 `+ 1`），文件改名 `app.py`；docstring 注明单入口。
- 前端 `vite.config.ts` proxy 目标改 `http://127.0.0.1:7861`；`README` 快速开始改单入口（`python app.py`）。
- GPU 机：部署 → 杀 7862 serve 与旧 7861 gradio → 起新 `app.py`（7861）→ 全端点 + root + legacy 已退场验证。
- 验收：7861 单进程挂 /api/* + SSE + dist（root 200 asset 新 hash）+ 7862 无服务 + pytest 全绿。

## Task 6 · 删 legacy + 清理 + 提交

**Files:** `legacy_app_gradio.py`（git rm）、`requirements.txt`（删 gradio）、README、`.superpowers/sdd/progress.md`

- GPU 机实机验证单入口**至少半天**（期间任何回归都还能双开对照）→ 确认后 `git rm legacy_app_gradio.py`；`requirements.txt` 删 `gradio>=4.44`，GPU 机 `uv pip uninstall gradio`（或重装 requirements，以 uv 输出为准）。
- 残留终审 grep：`gradio|legacy_app|7862|server_port + 1`（期望 CLEAN，docs/ 计划文件除外——历史文档不改写）。
- README：删双开描述/legacy 段；「目录」改 `app.py = FastAPI 单入口`。
- 记忆同步（ADR 024 状态行「Plan 1+2+3+4 全部落地」、05 ModelOpt 行 ≤6144B）+ 台账 + 全量回归 + 一次提交。
- 验收：pytest 全绿 + GPU 机 7861 全端点 200 + grep CLEAN + 记忆门禁过。

## 自检（计划自洽性，执行前核）

1. §7 层 3 三层应对：层 1（capabilities）/ 层 2（发射前差集校验）Plan 2 已落地 ⇒ 本计划只做层 3（逃生口）+ 前端 unmapped 标签 ✓
2. §8 C1/C2 两版的增量面 = 后端两字段 + 前端交互，不重构已有最小版 ✓
3. 估算器公式全部落在 `server/estimate.py` 文件头常量，UI 不拍公式（设计稿 §6 的「不做位宽切换按钮」遵守）✓
4. D2 单入口顺序：golden 拔依赖 → 改名 → 起 7861 → 验证 → 最后删 legacy（Task 5/6 分两步，中间可回滚：git mv 可逆）✓
5. 真跑量化/评测不在本计划（ADR 023），显存空闲时补——验证记录里如实记 ✓

## 执行选项

- **① inline（默认）**：按 m03282 指令直接主会话做，Task 1→6 顺序执行，每 Task 提交。
- **② subagent-driven**：worker 派工 + 主会话复跑 + 收窄 reviewer（Plan 2 模式，耗时风险高）。

## GPU 机验证记录（2026-10-14）

**单入口终态**：`python app.py` 挂 7861（`/api/*` 23 端点 + SSE + `web/dist` 五区），
legacy Gradio 已删、`serve.py` 退役、`requirements.txt` 去 `gradio>=4.44`、
`golden_adapter.py` 拔 `app.py` 依赖（改用 `webui/input_order.py`）。

**GPU 机实测**（chrisqian@192.168.124.39，`uv pip uninstall gradio` + 清残留空目录后）：
- `import gradio` → `ModuleNotFoundError: No module named 'gradio'`（彻底不可用）
- 7861 全端点真值 200：health / bootstrap / capabilities / POST probe / vocab /
  jobs / models/local / artifacts / estimate / eval/results；root 返
  `index-Cj1jTANf.js`（新 dist）
- 7862 → `code=000`（无服务，serve.py 已退役）

**回归**：`.venv/bin/python -m pytest -q` = **106 passed**；
`tsc --noEmit -p tsconfig.app.json` = 0 错。

**逃生口（层 3）**：`--verbose` 等 unmapped flag 经 extra_args 原样追加命令尾，
禁用字符 `;&|<>\n\t$\`` 触发 400 `bad_params`；`--definitely_not_a_flag` 走差集校验
400 `unknown_flags`（层 2 兜底仍生效）。

**估算器**：权重量 = hub 缓存目录 `du`（`source=cache`）→ HF API size（`hf_api`）
→ 未知（`unknown`）；量化后按 qformat 系数折算；KV 从 config.json 精确算或经验式
回退（`source` 追加 `+kv_estimate`）；与真实显存对比给 `fits` / `short_gib`。
GPU 机实测：已缓存模型 `Systran/faster-whisper-medium` 走 du 真实 1.5GB +
KV 经验式；未缓存走 unknown（代理不通，如实标注）。

**C1 产物库**：`/api/artifacts` 返回 `{dir, model, qformat, ts, size, state,
cmd}`（model/qformat 从 job.json.cmd 或 run.log 首行正则提取）；前端卡片 +
「生成 vLLM 启动命令」（复制）。

**C2 本机模型列表**：`/api/models/local` 扫 `hf_cache/*/models--*/*/snapshots/*`，
每条带 `revision`（`refs/main` 首行）+ `size`（du）；stub（无权重）标灰。

**真跑**：显存被 vllm serve 占（>24G），未跑真量化/评测（守 ADR 023 约束）。
浏览器项：GPU 机无 agent-browser，用后端端点直验替代（root asset hash + 全端点
HTTP 状态码），未做像素级 UI 截图验证。

# Plan 3 · React 前端（5 区）

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** 按设计稿 §6 把 `llm_model_quantization_dashboard.tsx` 的风格模版落实成真前端：`web/`（Vite + React + TS + Tailwind + recharts + lucide-react），侧栏 5 区（量化 / 评测 / 对配方 / 产物库 / 路径与环境），全部数据来自 Plan 2 的 22 个端点 + SSE，**零 mock**——模版里的 AWQ/GPTQ/GGUF 选择器、假模型列表、假进度、4 台假 server 一律不出现。验收：`npm run build` → `./deploy.sh` → GPU 机 7862 出真 UI。

**Architecture:** 三层，每区一个 view：
1. `web/src/api/`：`client.ts`（fetch 封装：base `/api`、`X-Api-Token` 从 localStorage、401 时触发 token 输入弹层、错误 envelope `{error:{code,message,detail}}` 统一抛出）+ `types.ts`（bootstrap/probe/vocab/caps/job/progress/rows 的 TS 类型，与 `server/schemas.py` 和 dataclass 字段一一对应）+ `endpoints.ts`（22 端点的薄封装）。
2. `web/src/hooks/`：`useBootstrap()`（首屏一把 + refetch）与 `useJobStream(jobId)`（EventSource 接 `/api/jobs/{id}/events`，log 增量进缓冲区、progress 覆盖、state 终态关流；断线带 `Last-Event-ID` 重连）。
3. `web/src/views/` 5 区 + `web/src/components/`（Sidebar 5 区导航 + 底部 GPU 状态卡、TopBar 区标题 + 版本徽章 + 显存指示、ProgressCard、LogPanel、Toast、ConfirmModal、JobsPanel）。

不引入 router（5 区 = state 切换）、不引入 redux。显存估算器、vLLM 命令模板、C1/C2 完整版 = Plan 4；本计划产物区只展示最小版 `/api/artifacts` 列表，量化区用 `/api/models/local` 做模型名自动补全。

**Tech Stack:** node（mac v26）/ Vite / React / TypeScript / Tailwind / recharts / lucide-react。GPU 机不装 node，只收 `web/dist`。

**设计依据:** 设计稿 `docs/superpowers/specs/2026-10-14-fastapi-react-ui-design.md` §6（前端）、§5.3/§5.5（API 与字段形状）、§5.6（token）、§9（验证策略）。风格基准 = 根目录 `llm_model_quantization_dashboard.tsx`：`bg-slate-950 text-slate-100`、`rounded-2xl`、`bg-slate-900/60 border border-slate-800 backdrop-blur-md` 卡片、violet/cyan 点缀、步骤数字圆标、mono 字体显数值、侧栏底部 GPU 状态卡、顶栏版本徽章 + 显存指示。

**不在本计划:** 显存估算器（Plan 4）、vLLM 启动命令模板与完整产物库（Plan 4）、删 legacy/单入口 7861（Plan 4 收尾）、任何量化命令行为改动（golden 已锁死）。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `web/package.json` `web/vite.config.ts` `web/tsconfig.json` `web/index.html` `web/src/styles.css` | 新建 | Vite 脚手架；dev proxy `/api → http://127.0.0.1:7862`；Tailwind 入口 |
| `web/src/main.tsx` `web/src/App.tsx` | 新建 | 挂载 + 5 区切换 + token 弹层 + Toast 宿主 |
| `web/src/api/client.ts` `types.ts` `endpoints.ts` | 新建 | fetch 封装 / 类型 / 22 端点薄封装 |
| `web/src/hooks/useBootstrap.ts` `useJobStream.ts` | 新建 | 首屏数据 / SSE 订阅 |
| `web/src/views/{Quantize,Eval,Recipe,Artifacts,Settings}View.tsx` | 新建 | 5 区 |
| `web/src/components/{Sidebar,TopBar,ProgressCard,LogPanel,JobsPanel,Toast,ConfirmModal}.tsx` | 新建 | 共用件 |
| `server/api.py` | **改（本计划唯一后端改动）** | token 中间件加 `?token=` query 回落（EventSource 不能带自定义 header） |
| `tests/server/test_static.py`（或新 `test_token.py`） | 加 2 用例 | token query 参数生效 / 错误 token 401 |
| `deploy.sh` | 改 | 删 `--exclude='web/dist/'`（dist 要随 rsync 上机；node_modules 仍排除） |
| `README.md` | 改 | 前端开发/部署流程（`npm run build` → deploy） |

---

## 全局约束

- **零 mock**：侧栏 GPU 卡、顶栏显存条、版本徽章全部读 `bootstrap.probe` / `capabilities` 真值；字段缺失时显示「—」，不编造。模版里的 `QUANT_ALGORITHMS`（AWQ/GPTQ/GGUF/EXL2）、`MODELS_LIST`、`defaultVRAM`、假 server 列表**一个都不移植**。
- **后端只改一处**：token 中间件支持 `?token=`（仅当 header 缺失时回落；query 里的 token 会进访问日志，文档里注明）。其余 22 端点只读。
- **每 Task 提交前**：`cd web && npm run build` 成功（含 `tsc --noEmit`）+ 全量 `pytest -q` 90 passed 不回归（后端改动时）。
- **SSE 行格式**（`server/jobs.py`）：`event: <log|progress|state>\nid: <日志字节偏移>\ndata: <json>\n\n`；state ∈ (done,failed,aborted) 后流终止。`useJobStream` 按此解析，断线重连带 `Last-Event-ID`。
- **端口约定**：dev 双进程（mac `serve.py` 7862 + vite 5173）；部署后 GPU 机 7862 = 新 UI，7861 legacy Gradio 保留对照（Plan 4 删）。
- **不真跑量化/评测**（ADR 023 不抢占）：GPU 被占时，UI 验证止于「命令预览逐字节对、job 202 链路、SSE 事件流、结果区空态」；真跑留待显存空闲。

---

## API 事实（已核对 `server/api.py` / `schemas.py` / `webui/*`，2026-10-14）

| 端点 | 请求 | 响应 |
|---|---|---|
| `GET /api/bootstrap` | — | `{paths:{repo,venv,task_dir,hf_cache}, net:{mode,proxy}, probe:HostStatus, vocab, capabilities, jobs:[JobSummary], defaults:dict}` |
| `POST /api/probe` | — | `{probe:HostStatus, vocab}` |
| `GET /api/vocab` | — | `{qformats:[str], kv:[[name,value]], datasets:[{key,desc}], combos:{k:desc}, source, error}` |
| `GET /api/capabilities` | — | `{version, commit, flags_all[], mapped[], unmapped[], missing[], error}` |
| `POST /api/preview` | `QuantizeParams`（20 字段，`extra=forbid`） | `{cmd, pretty, missing_flags:[], caps_error}` |
| `POST /api/quantize` | `QuantizeParams` | `202 {job_id, job:JobSummary, warn_gpu:bool}`；400 码：bad_params/unreachable/venv_missing/bad_model_path/unknown_flags |
| `POST /api/eval` | `EvalStart{quant,baseline,tp,port,batch,tasks[]}` | `202 {job_id, job}`；400：eval_env_missing/bad_params |
| `POST /api/venv` / `POST /api/eval-env` | — | `202 {job_id, job}`（task=None 包装线程，SSE 同样可用） |
| `GET /api/jobs` / `GET /api/jobs/{id}` | — | 列表 / `{job, state, log_tail, progress:ProgressReport, result_tables}` |
| `GET /api/jobs/{id}/events` | header `Last-Event-ID`（或本计划新增 `?token=`） | SSE：事件 log / progress / state（三事件） |
| `POST /api/jobs/{id}/abort` | — | 中止（终态 no-op） |
| `POST /api/recipe/diff` | `{job_id, official}` | `{summary, groups, diff, files}`；404 no_job / 400 bad_job |
| `POST /api/eval/baseline` | `{quant_path}` | `{baseline, message}`；400 bad_quant_path |
| `GET /api/eval/results` | — | `{rows:[模型,任务,指标,量化,基线,Δ], note, files}` |
| `GET/PUT /api/paths` | PUT: `{repo,venv,task_dir,hf_cache}` | `{paths}` / `{message}`（空字段回填默认） |
| `GET/PUT /api/net` · `POST /api/net/test` | PUT: `{mode,proxy}` | `{net}` / `{message}` / `{ok, detail}` |
| `GET /api/models/local` | — | `{models:[{id, size, …}], error?}`（C2 最小版） |
| `GET /api/artifacts` | — | `{artifacts:[{dir, model, ts, size, state}]}`（C1 最小版） |

`QuantizeParams` 20 字段：`model, trust_remote_code, qformat, recipe="", kv_cache_qformat="none", datasets:[str], dataset_sizes:{str:int}, custom_datasets:[[str|int]], calib_seq=512, calib_with_images=false, export_path="", batch_size=1, device="cuda", attn_implementation="sdpa", use_fsdp2=false, cpu_offload=false, low_memory_mode=false, gpu_max_mem_percentage=0.0, skip_generate=false, extra_args=""`。表单默认值先读 `bootstrap.defaults`（config.yaml `defaults:` 段），缺省再落 schema 默认；每数据集条数默认 1024（可逐行改，对应 `dataset_sizes`；自定义数据集行进 `custom_datasets`）。

`ProgressReport`：`stage_idx, stage, pct(0-100|null), bar, detail, item, done, error`。
`HostStatus`（probe）：`reachable, auth, gpu, gpu_util, gpu_mem, gpu_busy, repo_ok, venv_ok, uv_ok, disk_free, hf_cache, hf_direct_ok, hf_proxy_ok, proxy_note, error`。

---

## Task 0 — 脚手架 + 部署通道

- [ ] `cd web && npm create vite@latest . -- --template react-ts`（在已有空目录 `web/` 内）→ `npm i` → `npm i tailwindcss @tailwindcss/vite recharts lucide-react`（Tailwind v4 走 vite 插件，`styles.css` 里 `@import "tailwindcss"`）
- [ ] `vite.config.ts`：`server.proxy = {'/api': 'http://127.0.0.1:7862'}`；`build.outDir = 'dist'`
- [ ] `deploy.sh`：删 `--exclude='web/dist/'` 一行（dist 上机；`web/node_modules/` 保留排除）。`./deploy.sh` 干跑确认（mac 本机先 `npm run build` 出 dist）
- [ ] 验证：`npm run build` 出 `web/dist/index.html` + assets；GPU 机 `curl -s http://127.0.0.1:7862/ | head -3` = index.html（SPA catch-all 生效，不再是 no_dist JSON）
- [ ] 提交 `chore(web): Vite+React+TS+Tailwind 脚手架 + deploy.sh 同步 dist（Task 0）`

## Task 1 — api 层 + App 外壳

- [ ] `api/types.ts`：上表全部响应形状（HostStatus/Vocab/Capabilities/JobSummary/ProgressReport/QuantizeParams/…），字段名与 Python 侧逐字一致
- [ ] `api/client.ts`：`request<T>(path, init)`；401 `bad_token` → 派发 token 弹层事件；非 2xx 抛 `ApiError{code,message,detail}`；token 存 localStorage（键 `mow.token`）
- [ ] `api/endpoints.ts`：bootstrap/probe/vocab/capabilities/preview/quantize/eval/venv/evalEnv/jobs/jobDetail/abort/recipeDiff/evalBaseline/evalResults/paths/net/netTest/modelsLocal/artifacts 各一个函数
- [ ] `hooks/useBootstrap.ts`：挂载即取，暴露 `{data, error, reload}`
- [ ] `App.tsx`：侧栏（logo「ModelOpt Quant」+ 5 区导航 + 底部 GPU 卡：`probe.gpu` / `gpu_mem` / `gpu_util` / `gpu_busy` / `disk_free`，真实值）+ 顶栏（区标题 + `capabilities.version` 徽章 + 显存指示点）+ 5 区占位（每区一个「建设中」卡片）+ Toast 宿主 + token 弹层（401 时出现，输入存 localStorage 后重放当前请求）
- [ ] 验证：`npm run build` OK；mac 起 `serve.py` + `npm run dev`，5173 能渲染侧栏/顶栏，bootstrap 失败（mac 无 GPU 也没关系）时不白屏、显错误态；GPU 机 dist 下 bootstrap 真值渲染（`capabilities.version` = `0.48.0rc0-35-g254195350`）
- [ ] 提交 `feat(web): api 层 + App 外壳（侧栏/顶栏/GPU 卡/token 弹层）（Task 1）`

## Task 2 — SSE + 进度卡 + 日志面板（含后端 token query 回落）

- [ ] **后端**（唯一改动）：`server/api.py` token 中间件加回落——`header X-Api-Token` 缺失时读 `request.query_params.get("token")`；query 值同样要与 `CFG.server_token` 相等。新增 2 测试（query 正确 → 200；query 错误 → 401）。`pytest -q` 全绿（90 + 2）
- [ ] `hooks/useJobStream.ts`：`EventSource('/api/jobs/'+id+'/events' + (token ? '?token=*** : ''))`；onmessage 按 `event` 分流：log → 追加（保留上限 5000 行）、progress → 覆盖、state → 更新 + 终态 close；断线（readyState CONNECTING）自动重连，带 `Last-Event-ID`（记住最后收到的 id）
- [ ] `components/ProgressCard.tsx`：`ProgressReport` → 阶段名 + pct 进度条（pct 为 null 时显示「进行中 · detail」）+ `item`/`detail` 行 + error 红字；done 时绿色完成条
- [ ] `components/LogPanel.tsx`：等宽字体、自动滚底（用户上滚则暂停）、SSE 增量刷新
- [ ] `components/JobsPanel.tsx`：`bootstrap.jobs` 列表（kind/state/cmd_preview/started_at），点进详情（job_detail 快照 + 订阅该 job 的 SSE 续看）
- [ ] 验证：`npm run build` OK；GPU 机起一个**真 job**（`POST /api/venv`——venv 已存在会秒级早退走 `_done` 路径，正好验证终态）：UI 看到 log 事件流 + state=done + ProgressCard 完成态；`curl -N` 同 job 对比事件序列一致；断网重连带 `Last-Event-ID` 不重不漏（人工断 ssh 复现一次）
- [ ] 提交 `feat(web): SSE 订阅 + ProgressCard + LogPanel + JobsPanel；后端 token query 回落（Task 2）`

## Task 3 — 量化区（QuantizeView）

- [ ] 表单（对照 `QuantizeParams` 20 字段，分区卡片：①模型 ②量化 ③校准 ④导出/资源 ⑤高级）：
  - 模型：`model` 输入 + `<datalist>` 自动补全（`GET /api/models/local`）+ `trust_remote_code`
  - 量化：`qformat` 下拉（`vocab.qformats`）+ `recipe`（可空）+ `kv_cache_qformat`（`vocab.kv` 的 [name,value]）
  - 校准：`datasets` 多选（checkbox 列表，`vocab.datasets` {key,desc}）+ 每数据集条数行（默认 1024）+ `custom_datasets` 加行（名称+条数）+ `calib_seq` + `calib_with_images`
  - 导出/资源：`export_path` / `batch_size` / `device` / `gpu_max_mem_percentage`
  - 高级：`attn_implementation` / `use_fsdp2` / `cpu_offload` / `low_memory_mode` / `skip_generate` / `extra_args`（逃生口，提示「原样拼到命令尾，禁止 ; 与 &&」）
  - 默认值：`bootstrap.defaults` 覆盖 schema 默认
- [ ] 「命令预览」→ `POST /api/preview`：显示 `pretty`（pre 块 + 复制按钮）；`caps_error` 非空时黄色提示条；400 `unknown_flags` 显示 detail 里的 flag 名单
- [ ] 「开始量化」→ 先 preview 校验，再 `POST /api/quantize`；`warn_gpu=true` 弹 ConfirmModal（「GPU 当前被占用，继续可能与别的进程抢显存」）；202 后切到进度区（ProgressCard + LogPanel + 中止按钮）
- [ ] 模型名空 / 必选缺省时按钮禁用并提示
- [ ] 验证：`npm run build` OK；GPU 机 preview 输出与 `curl` 直发同参数逐字一致（含 golden 4 用例参数之一，flag 串对 `tests/golden/quantize_cmd.json` 剥离路径前缀后相等）；显存被占 ⇒ **不点开始**，记录「真跑待显存空闲」
- [ ] 提交 `feat(web): 量化区（表单 20 字段 + 预览 + 启动/中止 + warn_gpu 确认 + C2 补全）（Task 3）`

## Task 4 — 评测区（EvalView）

- [ ] 表单：`quant`（量化 checkpoint 路径）+ `baseline`（输入 + 「建议」按钮 → `POST /api/eval/baseline`）+ `tasks` 多选（11 个 EVAL_TASKS）+ 预设行（AA 标准 / 快速检查 / 量化敏感扩展，一键勾选对应 tasks）+ `tp` / `port` / `batch`
- [ ] 「运行评测」→ `POST /api/eval` → 进度区（同 Task 2 组件，`pkill_pattern=vllm serve` 的中止按钮带「会杀掉 vllm serve」提示）
- [ ] 结果区：`GET /api/eval/results` → `rows` 表格（[模型,任务,指标,量化,基线,Δ]）+ recharts 分组柱状（按 task 维度，量化 vs 基线；`Δ` 负值标红）；无结果时显示 `note` 空态；**雷达图仅真实维度 ≥3 时渲染，否则只出表格**（设计稿 §6）
- [ ] 验证：`npm run build` OK；GPU 机无结果时空态正确；有结果时表格与 `curl` 一致（若显存空闲且用户点头则真跑一次快速检查预设，否则记录「真跑待显存空闲」）
- [ ] 提交 `feat(web): 评测区（表单 + baseline 建议 + 运行 + 结果表/柱状图）（Task 4）`

## Task 5 — 对配方区 + 产物库区

- [ ] RecipeView：选 job（JobsPanel 里 state 非 running 的 quant job）+ `official` 官方 summary 路径输入 → `POST /api/recipe/diff`：`summary` 卡片 + `groups`/`files` 列表 + `diff` pre 块（空 official 时只出本产物 summary）
- [ ] ArtifactsView：`GET /api/artifacts` → 卡片列表（dir/model/ts/size/state 徽章，state=done 绿 / failed 红 / manual 灰）；顶部说明「vLLM 启动命令模板见 Plan 4」占位条
- [ ] 验证：`npm run build` OK；GPU 机两区渲染真数据（artifacts 空列表时显空态，不编造）
- [ ] 提交 `feat(web): 对配方区 + 产物库区（C1 最小版展示）（Task 5）`

## Task 6 — 路径与环境区（SettingsView）

- [ ] 路径卡：四路径输入（repo/venv/task_dir/hf_cache）→ `PUT /api/paths`（成功 toast + `probe` 重取）
- [ ] 环境卡：「创建量化 venv」→ `POST /api/venv`（进度条 = 该 job 的 SSE log 尾 + 完成判定 state=done；venv_ok 已 true 时按钮置灰「已就绪」）；「创建评测环境」→ `POST /api/eval-env` 同款
- [ ] 探活卡：`POST /api/probe` 按钮 + HostStatus 全字段明细表（reachable/gpu/gpu_util/gpu_mem/disk_free/repo_ok/venv_ok/uv_ok/hf_cache/hf_direct_ok/hf_proxy_ok/proxy_note/error），✗ 项红色高亮
- [ ] 网络卡：`mode` 三选（auto 先直连后代理 / direct 仅直连 / manual 手填）+ `proxy` 输入（占位示例 `http://192.168.124.24:7890`）→ `PUT /api/net`；「测试连通性」→ `POST /api/net/test` → `{ok, detail}` 绿/红提示
- [ ] 验证：`npm run build` OK；GPU 机改一个路径再改回（PUT→GET 一致）+ net/test 返回 `可用（HTTP 200/204，x.xs）` 真值
- [ ] 提交 `feat(web): 路径与环境区（路径/venv/eval-env/探活明细/代理）（Task 6）`

## Task 7 — 整体验证 + 文档 + 收尾

- [ ] 全量回归：mac `pytest -q` 92 passed；`import app; app._build_app()` BUILD OK（legacy 未动）
- [ ] `npm run build` → `./deploy.sh` → GPU 机 7862 五区逐一 curl + 真值核对（对照 Task 1-6 的验收表）；7861 legacy 同机对照仍可开（截图/记录双开状态）
- [ ] token 模式实测：`config.yaml` 开 `server.token` → 7862 全端点 401 直到填 token；SSE 走 `?token=`；关闭后恢复无感
- [ ] `README.md`：前端开发（`npm run dev` + serve.py）、部署（`npm run build` → `./deploy.sh`，dist 随 rsync 上机）、双开说明（7861 legacy / 7862 新 UI，单入口 7861 在 Plan 4）
- [ ] 计划文件末尾追加「GPU 机验证记录」（逐项打勾 + 未做项如实记：真跑量化/评测待显存空闲）
- [ ] 记忆同步（ADR 024 状态行、05 ModelOpt 行，≤6144B）+ 台账更新 + 一次提交 `docs: Plan 3 收尾 —— 整体验证记录 + README（Task 7）`

---

## 自检（计划自洽性，执行前核）

1. 设计稿 §6 的 5 区与本文 Task 1/3-6 一一对应 ✓（量化 Task 3 / 评测 Task 4 / 对配方 Task 5 / 产物库 Task 5 / 路径与环境 Task 6）
2. 模版 mock 元素处置：全部不移植（约束「零 mock」+ 各 Task 无 AWQ/GPTQ/假模型/假 server）✓
3. 后端唯一改动 = token query 回落（Task 2），其余 22 端点只读 ✓
4. 估算器/vLLM 模板/单入口/删 legacy 全部在「不在本计划」✓
5. SSE 重连语义与 `server/jobs.py` 的 id=日志偏移、终态 break 一致（Task 2 按此实现）✓

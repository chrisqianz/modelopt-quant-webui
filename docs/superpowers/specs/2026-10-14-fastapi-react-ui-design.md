# ModelOpt Quant WebUI —— 换掉 Gradio：FastAPI + React 改版设计

- 日期：2026-10-14
- 范围：UI 层重写（Gradio → FastAPI + React），并删除 SSH 远程执行模式
- 不动：`webui/` 里的业务逻辑（runner / progress / config / validate / vocab）
- 设计稿：`llm_model_quantization_dashboard.tsx`（1131L 静态稿，**只取风格，不做一比一复刻**）
- 前置决策：ADR 023（本项目立项）；本次改版落 ADR 024（待写）

---

## 1. 目标与非目标

### 目标
1. 拿到设计稿那套「侧栏 + 分区面板 + 状态条」的视觉与交互，Gradio 给不了
2. 现有功能全量平替（量化、评测、对配方、环境/网络），搬完删掉 Gradio 入口
3. 新增两项：量化产物库（C1）、本机 HF 缓存模型列表（C2）
4. 修掉两个长期难受的点：刷新页面跟丢任务；长请求把 UI 请求阻塞住

### 非目标
- 不做 AWQ / GPTQ / GGUF / EXL2 / SmoothQuant 算法选择器 —— 本项目只有 ModelOpt NVFP4 一条路，设计稿那排按钮是 mock
- 不做多机管理 —— WebUI 部署在哪台机器就管哪台（见 §3 决策 D3）
- 不做登录/多用户；不做一比一像素复刻
- 不改量化命令本身（`build_remote_cmd` 产出的命令串必须逐字节不变，见 §9）

---

## 2. 已核实事实（不是推测）

### 代码
| 事实 | 证据 |
|---|---|
| `gr.Tabs` 确实没有 `placement` 参数 | 实测 `gr.Tabs.__init__` 签名：`selected, overflow_behavior, visible, elem_id, elem_classes, render, key, preserved_by_key` |
| 现在的侧栏本来就是手工切的 | Radio + 4 个 `visible=False` 的 `gr.Column`，`nav.change` 在 `app.py:791` |
| 后端零 Gradio 依赖 | `webui/*.py` 无 `import gradio`，只有 `progress.py:109` 一句注释提到 |
| **本机模式早就写好了，而且是默认值** | `config.py:33-36` `MODE_LOCAL`/`MODE_REMOTE`；`HostConfig.is_local` `config.py:62`；`runner.py` 已有 8 处 `if host.is_local:`（L55/163/174/209/221/259/461/501）；网络配置分 `local_proxy`/`remote_proxy`（`runner.py:486`）；`app.py:739` 「执行位置」Radio 默认 `MODE_LOCAL` |
| 现在跑成远程只因 hosts.json 填了 ssh | `config.py:157`：ssh 非空即自动判为 remote |
| 刷新会丢任务 | 任务态在 `gr.State`（`task_state`/`eval_state`/`hosts_st`） |
| 长任务靠 UI 侧轮询 | `do_start` `app.py:994`、`do_eval` `app.py:1141` 是 generator，内部 `while True: ... time.sleep(5)` |

### 环境（GPU 机 chrisqian-ubuntu01 实测）
| 项 | 结果 |
|---|---|
| 直连 huggingface.co | `000` 不通 |
| 直连 hf-mirror.com | `200` 通 |
| 走 mac 代理 `http://192.168.124.24:7890` | `204` 通（代理允许局域网，GPU 机可用） |
| Python / node | 3.10.12 / v20.20.2 都在 |
| `~/Model-Optimizer` | 在 |
| `~/.venvs/modelopt` | **不在**（首次使用需点「创建 venv」） |
| `~/.cache/huggingface/hub` | 存在，标准 `models--org--name` 布局；部分条目是 stub（如 `models--apolo13x--Qwen3.5-27B-NVFP4` 只有 `refs`，12K） |
| `~/quant-webui-tasks` | 不存在（还没跑过任务） |
| `~/modelopt-quant-webui` | **不存在** —— WebUI 还没上过 GPU 机，删 SSH 后必须先有部署通道（§10 第 0 步） |
| Model-Optimizer 版本 | `0.48.0rc0-35-g254195350`，最后一次提交 **2026-10-13（探测的前一天）** —— 上游是活跃移动靶 |
| `examples/hf_ptq/hf_ptq.py` | 存在，共 **35** 个 CLI flag；`build_remote_cmd` 只映射 **18** 个，这 18 个在新版全部仍存在（今天不报错），**17 个功能界面上根本不存在** |
| `modelopt_recipes/` | 存在 —— `vocab.py` 动态扫词表这条路本来就是版本自适应的 ✅ |

---

## 3. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 换掉 Gradio，接受 FastAPI + React 双栈 | 目的就是要设计稿那套视觉；接受多一个 node 构建步骤 |
| D2 | 单进程单入口 | `python app.py` 起 uvicorn，端口仍读 `config.yaml`（7861），同端口挂 `/api/*` + SSE + 静态托管 `web/dist`。node 只在 mac 开发用，GPU 机只要 python（`dist/` 随目录 rsync，不破 ADR 023 的可移植性） |
| D3 | **删除 SSH 远程模式** | 用户判断：部署在任意服务器上都能直接取数据，不必再实现一遍 SSH。且本机模式本就是默认值，这是减法不是加法 |
| D4 | 范围取 C 档：全量平替 + C1 产物库 + C2 本机缓存列表 | A 档会长期并存两个入口、两份配置；C 的两项数据源现在都是真的 |
| D5 | 日志/进度用 SSE，不用 WebSocket | 单向推送，浏览器自带重连，少一套协议，单端口够用 |
| D6 | 任务态从 `gr.State` 挪到服务端 registry + 落盘 | 刷新不丢、后端重启可续、天然支持多任务并行 |
| D7 | 设计稿只取风格 | 显存条/评测图/雷达图全部用真实数据；雷达图只在真实维度 ≥3 时才画 |
| D8 | 现 Gradio 版改名 `legacy_app_gradio.py` 保留到搬迁验证完成 | 双开对照 + 当 golden 来源，之后删除 |
| D9 | 运行用 venv 建在**项目目录内**（`<项目>/.venvs/modelopt`），不再放 `~/.venvs/` | 用户要求：别往 home 根目录塞东西；venv 与项目同目录，删项目即清干净。**代价**：部署必须 `--exclude='.venvs/'`，否则 `--delete` 会把 GPU 机上装好的 venv 删了（§5.2） |
| D10 | 与 Model-Optimizer 的版本耦合用三层应对：能力探测 → 发射前校验 → 附加参数逃生口 | 上游一天一变、flag 只增不减，靠「跟着改代码」追不上，得靠结构。详见 §7 |

---

## 4. 架构与目录

```
modelopt-quant-webui/
├─ app.py                 # 入口不变：起 uvicorn；dist/ 缺失时明确报错「先 npm run build」，不白屏
├─ server/                # 新，替代 app.py:526-1254 那 730 行 UI 代码
│  ├─ __init__.py
│  ├─ api.py              # 路由，与现有 27 个 handler 一一对应
│  ├─ jobs.py             # job registry + 后台 poller + SSE 扇出
│  ├─ schemas.py          # pydantic 模型，直接吐 dataclass 字段，不再拼 HTML
│  ├─ capabilities.py     # 上游版本 + hf_ptq flag 探测与映射表（§7）
│  └─ static.py           # 托管 web/dist + SPA fallback
├─ webui/                 # 业务逻辑不动，只删 remote 分支
├─ web/                   # 新：Vite + React + TS + Tailwind + recharts
│  ├─ src/{main.tsx, App.tsx, api/, hooks/, views/, components/}
│  └─ dist/               # 构建产物，随目录一起 rsync 到 GPU 机
├─ deploy.sh              # rsync 到 GPU 机，含必需的 --exclude（§5.2）
├─ legacy_app_gradio.py   # 现 app.py 改名留档（对照用，验证完删）
├─ config.yaml            # 新增 local: 段
├─ hosts.json             # 删除
└─ tests/                 # golden 比对 + 纯函数单测（不碰 GPU）
```

部署形态：仓库 rsync 到 GPU 机 → GPU 机上 `python app.py` → 浏览器开 `http://192.168.124.39:7861`。mac 只承担开发（改代码 + `npm run build` + rsync）。

新增依赖：后端 `fastapi` `uvicorn` `pydantic>=2`（进 `requirements.txt`）；前端 `vite` `react` `typescript` `tailwindcss` `recharts`（只在 `web/package.json`，GPU 机不装）。

---

## 5. 后端改动

### 5.1 删除清单（D3 带来的减法）
- 9 个 SSH 函数共 192L：`_ssh_base`(9) `ensure_master`(34) `check_connection`(22) `run_remote`(20) `ensure_hf_tunnel`(26) `proxy_url_for`(24) `drop_master`(10) `test_proxy`(24) `vocab.fetch_remote_qformats`(23)
- `runner.py` 8 处 `if host.is_local:` 分支拉平成单路径；`runner.py` 92 处、`app.py` 28 处 ssh/隧道关键词大部分随之消失
- `HostConfig` 的 `ssh` / `ssh_port` / `password` 字段；`MODE_REMOTE`、`EXEC_MODES` 两条选一
- UI 侧：SSH 地址框、端口框、密码框、「重连/隧道」按钮、「执行位置」Radio
- `hosts.json` 及其 CRUD（`_apply_hosts` `app.py:499` 等）

`_ensure_hf_tunnel` 原本解决的 HF 网络问题，实测用一个代理地址输入框即可覆盖（§2 环境表），符合 ADR 023 第 5 条「代理地址一律手填、不猜拓扑」。

### 5.2 配置形态
`hosts.json` → `config.yaml` 一段：

```yaml
local:
  repo: ~/Model-Optimizer          # 原 remote_repo 改名
  venv: .venvs/modelopt            # 相对项目根，不再往 ~ 里塞（D9）
  task_dir: ~/quant-webui-tasks    # 产物 GB 级，故意留在项目外，不随 rsync 走
  hf_cache: ~/.cache/huggingface
```

相对路径按「装着 `config.yaml` 的那个目录」解析（`config.py` 新增 `resolve_relative()`，与 `expanduser` 串用）。`server.host` / `server.port` / `network.*` / `defaults.*` 保持原样。首次启动若 `local:` 缺省，用现有 `DEFAULT_*` 常量兜底并写回。

部署命令固定进 `deploy.sh`：

```
rsync -a --delete --exclude='.venv/' --exclude='.venvs/' \
     --exclude='__pycache__/' ./ chrisqian@192.168.124.39:~/modelopt-quant-webui/
```

`--exclude='.venvs/'` 是硬要求：venv 挪进项目之后，漏掉它就会让 `--delete` 把 GPU 机上 5–15 分钟装出来的 venv 删掉（mac 侧不存在这个目录）。`task_dir` 留在项目外，正是为了避免 GB 级产物被拖进这条 rsync。

### 5.3 API（无 host 维度）

| 方法 路径 | 来源 handler | 说明 |
|---|---|---|
| `GET /api/bootstrap` | `initial_probe` L973 | 首屏一把：路径配置、网络模式、词表、进度位、活跃 job 列表 |
| `POST /api/probe` | `do_refresh` L941 | 探活 + 重扫词表 → `HostStatus` 原始字段 + qformat/kv 选项 |
| `GET /api/vocab` | `vocab.*` | 配方/量化格式/KV/数据集词表（本地扫 `modelopt_recipes`，离线快照兜底） |
| `GET /api/capabilities` | 新（§7） | Model-Optimizer 版本 + `hf_ptq.py` 真实 flag 集 + 已映射/未映射/已失效分类 |
| `POST /api/preview` | `refresh_preview` L983 | `build_params`+`build_remote_cmd` → 命令串 |
| `POST /api/quantize` | `do_start` L994 | `202 {job_id}`，后台执行 |
| `POST /api/eval` | `do_eval` L1141 | `202 {job_id}` |
| `POST /api/venv` | `do_venv` L1055 | `202 {job_id}`（建 venv 是分钟级，Gradio 版把请求阻塞住了） |
| `POST /api/eval-env` | `do_eval_env` L1070 | `202 {job_id}` |
| `GET /api/jobs` | — | 活跃 + 历史 job（刷新续上的关键） |
| `GET /api/jobs/{id}` | — | 单 job 状态快照 |
| `POST /api/jobs/{id}/abort` | `do_abort` L1049 / `do_ev_abort` L1180 | |
| `GET /api/jobs/{id}/events` | generator 里的 `while True` | SSE |
| `POST /api/recipe/diff` | `do_validate` L1086 | `{summary, groups, diff, files}` |
| `POST /api/eval/baseline` | `do_fetch_baseline` L1114 | 建议 baseline |
| `GET /api/eval/results` | `do_fetch_results` L1185 | 结构化分数行，表与图共用一份数据 |
| `GET /api/models/local` | 新（C2） | 本机 HF 缓存模型列表 |
| `GET /api/artifacts` | 新（C1） | 量化产物列表 |
| `GET /api/paths` / `PUT /api/paths` | `save_host` L841 拆分 | 路径配置读写 |
| `GET /api/net` / `PUT /api/net` / `POST /api/net/test` | `save_net_cfg` L926 / `test_net` L931 | 代理配置与连通性测试 |

不占接口的纯前端联动（Gradio 版每次点一下都发请求，新版在 React 状态里做）：`sizes_rows` L990、`toggle_ssh` L837（随 SSH 一起消失）、`do_ev_preset` L1136、`net_of` L921、`combo_dd` 联动。

### 5.4 jobs.py
- registry：`dict[job_id, Job]`，Job 包 `Task` + 日志 offset + 订阅者队列 + 线程
- 后台 poller 沿用 `do_start` 的逻辑（`poll_task` + `read_log` + `progress.parse`，5s 一轮），只是搬到线程里；一个 job 一个 poller，多标签页共享（扇出）
- SSE 三种事件：`log`（增量）、`progress`（`ProgressReport` JSON）、`state`（running/done/failed/aborted）；每条带 id = 日志 offset，断线用 `Last-Event-ID` 续传，不重不漏
- 落盘 `task_dir/<job_id>/job.json`（pid、log_path、cmd、state、export_dir），启动时扫一遍，pid 存活检测重新挂上还在跑的进程
- 同一台机器重复提交量化任务：不禁止，返回「GPU 可能打架」的确认标记，前端弹二次确认（ADR 023 不抢占约束靠人确认落地）

### 5.5 schemas 直接吐结构化字段
- `ProgressReport`（`progress.py:93`）：`stage_idx, stage, pct, detail, item, done, error` → 前端画进度卡；`bar`/`markdown()`/`_prog_html` `app.py:366`/`_STAGE_CSS` 全部废弃
- `HostStatus`（`runner.py:283`）：`reachable, auth, gpu, gpu_util, gpu_mem, gpu_busy, repo_ok, venv_ok, uv_ok, disk_free, hf_cache, hf_direct_ok, hf_proxy_ok, proxy_note, error` 原样输出；`as_dict()` 那套中文键（给 Gradio DataFrame 用的）不再给前端用
- `Task`（`runner.py:679`）：`task_id, pid, state, log_path, started_at`（`host` 字段随 SSH 一起消失）
- 错误统一 `{error:{code,message,detail}}`；HTTP 状态码用对（400 参数、409 冲突、502 远端失败）

### 5.6 安全
SSH 密码不再存在，凭据面归零。但 UI 仍绑 `0.0.0.0` 且无鉴权 —— 与 Gradio 版相同，不是回退；不过新版能列目录、改路径、建 venv、中止任务，暴露面比原来大。方案：`config.yaml` 加可选 `server.token`，非空时中间件校验 `X-Api-Token`（前端首次输入后存 localStorage），默认空 = 关闭，保持局域网便利。默认不启用，需要时一行配置打开。

---

## 6. 前端

栈：Vite + React + TypeScript + Tailwind + recharts + lucide-react（设计稿用的就是 lucide 图标）。不用 redux —— 每个 view 一个 hook + 一个 `api/` 封装 + 一个 `useJobStream(jobId)` SSE hook 足够。

侧栏 5 区：

| 区 | 内容 | 数据 |
|---|---|---|
| 量化 | 模型名（C2 自动补全）、qformat 下拉、校准数据集**多选**、参数、命令预览、开始/中止、进度卡、日志 | `/api/probe`、`/api/vocab`、`/api/preview`、`/api/quantize`、SSE |
| 评测 | lm_eval 参数、baseline 建议、运行、进度/日志、结果表 + 图 | `/api/eval`、`/api/eval/results`、SSE |
| 对配方 | 选 export 目录 → 与官方配方 diff | `/api/recipe/diff` |
| 产物库（C1） | 产物卡片 + 生成 vLLM 启动命令并复制 | `/api/artifacts` |
| 路径与环境 | 四个路径 + 创建 venv/评测环境 + 代理（直连/hf-mirror/手填）+ 测试 + 探活明细 | `/api/paths`、`/api/net`、`/api/probe` |

顶栏显存条：`probe` 的 `gpu`、`gpu_mem`、`gpu_util`、`gpu_busy`、`disk_free` 真实值 —— 32GB 容量与当前占用，不用设计稿的 `defaultVRAM` mock。

显存估算器（设计稿 L1015 是右侧滑面板；原公式 `modelSize*0.15*(ctx/2048)*batch` L1025 是拍的经验值）改为：权重量 = 本机已缓存则 `du` 缓存目录、未缓存则 HF API 取文件字节数；NVFP4 按权重量固定折算（不做位宽切换那排按钮）；KV 项按 `config.json` 的层数/头维度算，取不到则退回经验公式并标注「估算」。输出直接对着真实显存总量给「放得下 / 还差 X GB」。

图表：评测结果用 recharts 柱状/折线（真实分数）；雷达图仅在真实维度 ≥3 时渲染，否则退回表格。

---

## 7. 与 Model-Optimizer 的版本耦合（长期项目，登记在记忆 `00`）

本项目依赖 Model-Optimizer，而它是 editable 安装（`ensure_venv` `runner.py:554` 跑的是 `uv pip install -e '.[hf]'`）—— 在 `~/Model-Optimizer` 里 `git pull` 一下就换行为，WebUI 一行都不用改。升级便宜是好事，代价是**上游新功能界面看不见、旗标还可能改名**。探测时的实情：上游 `0.48.0rc0-35-g254195350`（提交日期是探测前一天），`hf_ptq.py` 有 35 个 flag，`build_remote_cmd` `app.py:131` 只映射 18 个，剩下 17 个 —— `--auto_quantize_checkpoint`、`--kv_auto_quantize_checkpoint`、`--export_fmt`、`--cast_mxfp4_to_nvfp4`、`--moe_calib_experts_ratio`、`--max_gpu_memory_gb`、`--max_cpu_memory_gb`、`--offload_folder`、`--sparsity_fmt`、`--specdec_offline_dataset`、`--vllm_fakequant_export`、`--use_seq_device_map`、`--inference_pipeline_parallel`、`--inference_tensor_parallel`、`--verbose`、`--model`、`--awq_block_size` —— 在界面上根本不存在。

耦合面其实只有五处，逐个可测：

| 耦合点 | 现状 | 应对 |
|---|---|---|
| 入口路径 `examples/hf_ptq/hf_ptq.py` | 硬编码在 `build_remote_cmd` | 探活时 `test -f`，缺失就直接说「上游挪了目录」 |
| 18 个 CLI flag | 硬编码拼串 | 能力探测 + 发射前校验 |
| 词表（qformat / recipe / dataset） | `vocab.py` 动态扫 `modelopt_recipes/` | 已经自适应，不动 |
| 日志文本 → 阶段与完成标记 | `progress.parse`、`Task.done_marker` | 失配则降级（见下） |
| 评测栈版本 | `vllm>=0.10.1`、`lm_eval[ifeval]>=0.4.12,<0.5` 钉在 `ensure_eval_env` | 显示在「路径与环境」，升不升由人点 |

三层应对：

1. **能力探测** `GET /api/capabilities`：版本取 `git -C <repo> describe --tags` + `rev-parse --short HEAD`；flag 集合对 `hf_ptq.py` 源码做静态抽取（提 `add_argument`），**不跑 `--help`** —— 那要 import torch，探活不该等 venv。与 `server/capabilities.py` 里一份 `FLAG_UI` 映射表比对，输出 `{version, commit, flags_all[], mapped[], unmapped[], missing[]}`。每个 job 元数据记下当时的 version/commit，事后能回答「这个产物是哪版上游跑出来的」。
2. **发射前校验**：`POST /api/preview` 和 `POST /api/quantize` 先把准备发出的 flag 与 `flags_all` 求差，有差就 400 并列出「本版本没有的参数」。**上游改名的后果从「跑 5 分钟才死」变成「点下按钮就告诉你」。**
3. **附加参数逃生口**（最关键）：量化区留一个「附加参数」输入框，内容原样追加到命令尾部（`shlex` 校验，禁 `;` `&&` 等）。再把 `unmapped[]` 渲染成可点标签，点一下就填进框。于是**上游新功能落地当天就能用，不用等 WebUI 发版** —— 这是把「长期跟涨」从无穷体力活变成有界结构的关键。

日志耦合单独兜底：`progress.parse` 认不出阶段时显示「未识别阶段 + 原始日志」，不画假进度；任务是否结束**以进程退出为准**，不依赖 `done_marker` 那句英文（上游改文案是常事）。

长期目标的形状（不是「上游每更新一次就改一次代码」）：
- 日常：探测 → 页面上看见 unmapped → 用附加参数先跑起来
- 上游 minor 版：从 `unmapped[]` 里挑值得做控件的正式表单化，顺手补 golden
- 每次跟涨在 ADR 记一行：上游版本 + 新接入控件

---

## 8. 新增功能

### C1 产物库
- 导出目录的权威来源是任务日志：`Task.done_marker = "Quantized model exported to"`，`validate.fetch_remote_export`（`validate.py:54`）本来就是「从日志定位导出目录」的逻辑
- 因此产物来源两路合并：① 每个 job 记录里解析出的 `export_dir`；② 扫 `local.task_dir` 下的任务目录兜底（覆盖 UI 之外手动跑的任务）
- 每张卡：模型名、时间、qformat、大小（`du`）、路径；按钮「生成 vLLM 启动命令」纯前端模板拼串（设计稿 L944 也是前端行为），不占接口

### C2 本机缓存模型列表
- 扫 `local.hf_cache/hub/models--org--name` → 还原 `org/name`，附 `du` 大小与 snapshot revision
- stub 条目（只有 `refs`、几 KB）要如实标注，不能显示成「已下载 12K」
- 点一下把 model id 填进量化区输入框，替代手打；同时回答「这台机器已经有什么」

---

## 9. 验证策略

`legacy_app_gradio.py` 在 7862 端口并行跑，同一台机器同一份配置，双开对照。等价性用 golden 证明，不靠眼睛：

1. **`build_remote_cmd`（`app.py:131`）新旧产出的命令串逐字节相同** —— 量化到底跑什么全在这条串里，它相等即后端行为未变
2. `build_eval_script`（`app.py:212`）、`_result_tables`（`app.py:307`）、`progress.parse` —— 同一份固定日志/结果样本喂进去比结构化输出
3. 每个 endpoint 一条 curl 冒烟 + 一条「与 Gradio 行为一致」的人工勾选项，写进实现计划每个任务的验收里
4. mac 无 CUDA → 只代码验证（ADR 023 不抢占 GPU）：单测覆盖纯函数与临时目录；凡碰 GPU 的验证在 5090 上做

---

## 10. 实现顺序（高层，细节留给 plan）

0. 部署通道：`deploy.sh`（§5.2 那条 rsync）+ GPU 机首次 `~/modelopt-quant-webui`，**先确认 Gradio 版在 GPU 机上跑得通** —— 否则后面所有「与旧版对照」都没有对照组
1. `config.yaml` 增 `local:` 段 + `webui/config.py` 删 remote 分支 + 删除清单（§5.1）→ 旧 Gradio UI 在纯本地模式下仍可用（先证明减法没减坏）
2. `server/` 骨架：bootstrap、paths、net、probe、vocab、preview、**capabilities**，curl 全通，静态目录未建时报错信息
3. `jobs.py` + SSE + registry 落盘，量化主链路跑通（真实 GPU 机跑一次小样）
4. `web/` 脚手架 + 侧栏框架 + 量化区（含显存条、进度卡、日志）
5. 评测区 + 对配方区（含 golden 比对全部通过）
6. C1 产物库、C2 缓存列表、显存估算器、附加参数逃生口 + unmapped flag 提示（§7）
7. 删 `legacy_app_gradio.py`、删 `hosts.json` 相关残留、更新 README/AGENTS.md、写 ADR 024

---

## 11. 风险与未决

| 风险 | 处置 |
|---|---|
| 删 remote 分支时误伤本地路径（8 处分支交错） | 第 1 步单独做，且以「Gradio 版在本地模式下功能完好」为验收，减法与改版分开验证 |
| HF 代理地址变了（mac 换网络 → IP 变） | UI 手填 + 测试按钮，不硬编码；错误信息直接提示去「路径与环境」改 |
| `~/.venvs/modelopt` 当前不存在 | 首次使用需点「创建 venv」，UI 明确引导（`venv_ok=false` 时高亮） |
| 无鉴权 + 绑 0.0.0.0 | `server.token` 可选开关，默认关（§5.6） |
| 上游一天一变（探测时 `0.48.0rc0-35`，提交日期是探测前一天） | §7 三层应对；已 `git init`，基线 `6291c21`，改版可回滚 |
| `hf_ptq.py` 入口路径或 flag 被上游改名 | 能力探测 + 发射前校验，点按钮时就报「本版本无此参数」，而不是跑 5 分钟才死 |
| 上游改日志文案导致 `progress.parse` / `done_marker` 失配 | 进度卡降级为「未识别阶段 + 原始日志」；任务结束以进程退出判定，不靠 marker 字符串 |

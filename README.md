# ModelOpt 量化 WebUI

在浏览器里驱动 [Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
跑 PTQ 量化与「量化 vs 基线」评测，全程看真实日志、真实进度。

WebUI **跑在有 GPU 的机器上**，浏览器从局域网打开。没有「远程执行」这一说：
量化命令、venv 创建、评测环境、词表扫描，全部发生在 WebUI 所在的这台机器上。

**单入口（端口 7861）**：`app.py`，一个进程同时挂 FastAPI 后端（`/api/*` +
SSE）、`web/dist` 静态前端（五个功能区：模型量化 / 性能与精度评测 / 配方对比 /
产物库 / 路径与环境）。

> 历史：早期是「React 7862（serve.py）+ Gradio legacy 7861」双开对照，已合并
> 为单入口 7861；legacy Gradio 已退役（`legacy_app_gradio.py`，本版本删除）。

## 快速开始

### 1. 部署到 GPU 机

```bash
./deploy.sh   # rsync 到 GPU 机（可用环境变量覆盖目标）
```

> `--exclude='.venvs/'` 是硬要求：漏了它 `--delete` 会把 GPU 机上装好的几 GB
> 量化 venv 删掉。

### 2. 起服务

```bash
ssh <gpu机>
cd modelopt-quant-webui
# 服务端用一个轻量 venv（fastapi / uvicorn / pydantic）
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt

# 先有 web/dist（npm run build → deploy.sh 已带），再起单入口
setsid nohup .venv/bin/python app.py >/tmp/moweb.log 2>&1 &
# 浏览器打开 http://<gpu机ip>:7861
```

> GPU 机没装 `python3-venv`/`ensurepip`，一律用 `uv` 建 venv、`uv pip` 装包。
> 静态目录在 import 时固化：换 `web/dist` 后必须重启进程才生效。

### 3. 建量化 venv（一次性，5–15 分钟）

点状态栏的「创建 venv」——在 `local.venv`（默认项目内 `.venvs/modelopt`）
里 `uv pip install -e '<ModelOpt 仓库>[hf]'`，日志尾部出现 `VENV_DONE` 即完成。

## 路径配置

全部写在 `config.yaml` 的 `local:` 段（相对路径按 config.yaml 所在目录解析）：

| 键 | 默认 | 说明 |
|---|---|---|
| `repo` | `~/Model-Optimizer` | Model-Optimizer 仓库，`hf_ptq.py` 在 `examples/hf_ptq/` 下 |
| `venv` | `.venvs/modelopt` | 量化用虚拟环境（**项目内**，跟着仓库走、随 deploy.sh 部署） |
| `task_dir` | `~/quant-webui-tasks` | 产物与日志；故意在仓库外，免得被 rsync 拖走 |
| `hf_cache` | `~/.cache/huggingface` | HF 权重缓存 |

UI 的「运行环境」面板里改这四行并保存会写回 `config.yaml`（保存会抹掉
config.yaml 的注释）。

## HuggingFace 网络

`config.yaml` 的 `network:` 段。GPU 机常常直连 huggingface.co 不通：

- `mode: auto` — 先试直连，不通再走 `proxy`
- `mode: manual` — 强制走 `proxy`
- `mode: direct` — 不用代理

`proxy` 填形如 `http://192.168.x.x:7890` 的代理地址（本机代理写
`http://127.0.0.1:7890`，局域网其他设备访问 GPU 机时写那台代理机的 IP）。
UI「HuggingFace 网络」面板可切换并即时测连通性。

## 功能

- **量化**：模型（HF ID 或本机路径）→ 配方 / qformat + KV qformat →
  校准数据集（内置勾选 + 自定义行，逐行独立 calib_size）→ 高级参数 →
  命令预览 → 启动 / 实时日志 → 结果校验（`hf_quant_config.json` 对配方 diff）。
  词表（qformat / KV）运行时从仓库 `modelopt_recipes` 真扫，ModelOpt 升级后
  自动出现新选项；扫不到时回落到内置快照并标注。
- **评测**：vLLM 部署量化模型（可配基线对照）→ lm-eval 跑任务 →
  量化 vs 基线指标表。
- **显存保护**：探测到 GPU 被占（显存占用 > 24G）时启动前明确提示，
  不抢占正在用的 GPU。

## 前端（web/）

React 19 + Vite + Tailwind v4 + recharts，构建产物在 `web/dist/`，
由 `app.py`（`server/static.py` 的 SPA catch-all）托管（未匹配路由回
`index.html`，`/api/*` 绝不回退）。

```bash
cd web
npm run dev        # 开发：vite dev server，/api 代理到 http://127.0.0.1:7861
npx tsc --noEmit -p tsconfig.app.json   # 类型检查（根 tsconfig 是 solution 型，勿用）
npm run build      # 构建 dist
```

部署：`npm run build` 后 `./deploy.sh`（dist 随 rsync 上机），再**重启 app.py**。

接口层：`src/api/`（types 与 `server/schemas.py` 逐字段对齐；401 触发 token
弹层）。若开了 `server.token`（config.yaml `server:` 段），所有
`/api/*` 要求 `X-Api-Token` header；SSE 不能带 header，连接走 `?token=` 回落，
token 会进访问日志。默认空 token ⇒ 本机无感。

## 测试

`tests/golden/` 是量化/评测命令串的 golden 快照（路径钉死成
`/home/tester/...`，与跑测试的机器无关）。

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q
```

改命令拼接逻辑后若行为有意变化：`-m pytest --update-golden` 重生成并人眼过一遍。

## 目录

```
app.py            单入口（7861）：起 uvicorn 挂 server.api:app（/api/* + SSE + web/dist）
server/schemas.py 请求/响应模型（pydantic，与前端 src/api/types.ts 对齐）
server/api.py     23 端点 + token 中间件（header / ?token= 回落）
server/jobs.py    job 注册表：生命周期 / 日志广播 / SSE 订阅 / 落盘重载
server/capabilities.py  ModelOpt 版本 + hf_ptq.py flag 静态抽取
server/estimate.py 显存估算（权重量 / 量化折算 / KV / 真实显存对比）
server/static.py  web/dist 托管（缺失时明确报错，不白屏）
web/              React 前端（src/api 接口层 / src/views 五区 / src/components）
webui/config.py   配置（server / local / network / defaults）
webui/runner.py   本机执行：探测 / venv / 代理 / 任务生命周期
webui/vocab.py    词表：数据集 / qformat / KV（真扫 + 快照兜底）
webui/validate.py 结果校验：导出目录扫描 + hf_quant_config 对配方
webui/progress.py 日志解析 → 阶段进度
deploy.sh         rsync 部署通道
tests/            golden 快照安全网
```

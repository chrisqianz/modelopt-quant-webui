# ModelOpt 量化 WebUI

在浏览器里驱动 [Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
跑 PTQ 量化与「量化 vs 基线」评测，全程看真实日志、真实进度。

WebUI **跑在有 GPU 的机器上**（Gradio，端口 7861），浏览器从局域网打开。
没有「远程执行」这一说：量化命令、venv 创建、评测环境、词表扫描，全部发生在
WebUI 所在的这台机器上。

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
# UI 用一个轻量 venv（gradio）
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app.py
# 浏览器打开 http://<gpu机ip>:7861
```

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
app.py            Gradio UI 层（参数 → 命令串 → 任务 → 日志/结果）
webui/config.py   配置（server / local / network / defaults）
webui/runner.py   本机执行：探测 / venv / 代理 / 任务生命周期
webui/vocab.py    词表：数据集 / qformat / KV（真扫 + 快照兜底）
webui/validate.py 结果校验：导出目录扫描 + hf_quant_config 对配方
webui/progress.py 日志解析 → 阶段进度
deploy.sh         rsync 部署通道
tests/            golden 快照安全网
```

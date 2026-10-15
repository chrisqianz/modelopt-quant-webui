# ui-verify — 像素级 UI 截图验证

## 基准约定（换机/重建基线时必须遵守）

- 视口 **1920×1080**，`deviceScaleFactor=1`（`set viewport 1920 1080`）
- URL `http://localhost:7861/`（app.py 单入口，serve 需先启动）
- 逐区截图前等 2s 让 bootstrap/SSE 落定；设置区、量化区内容高于视口，需滚动后补截
- 应用内滚动容器是 `main`（`h-screen` 布局，document 本身不滚）；
  agent-browser 的 `scroll main down` 可能静默 no-op，用 eval 兜底：
  `document.querySelector('main').scrollTo({top: main.scrollHeight, behavior:'instant'})`
- 截图命令：`agent-browser --session quant-verify screenshot --full <path>`（full=整页，此处等于视口）

## 捕获清单

| 文件 | 内容 |
|---|---|
| `01-quantize.png` / `01-quantize-bottom.png` | 量化区上/下（数据集列表、导出参数、高级参数、任务区） |
| `02-eval.png` | 评测区（任务分组、服务参数、运行评测、结果卡） |
| `03-recipe.png` | 配方对照（空 job 态） |
| `04-artifacts.png` | 模型仓库（空 task_dir 态） |
| `05-settings-top.png` / `05-settings-mid.png` / `05-settings-bottom.png` | 设置区上/中/下（路径、环境、探测明细、HF 网络；内容仅高约 211px 时 mid=bottom） |

## 脚本

- `analyze.py` — 尺寸一致性、白屏率、整块近白/纯黑检测、侧栏/内容/顶栏平均亮度
  `.venv/bin/python ui-verify/analyze.py`
- `diff.py` — 与 `baseline/` 逐像素对比（亮度差阈值 8/255，>2% 像素超阈值判 FAIL，
  输出 16× 差异热力图 `diff-<name>.png`）
  - 建立基线：截图放 `current/` 后跑 `--save-baseline`
  - 对比：`.venv/bin/python ui-verify/diff.py`（退出码 0=PASS 1=FAIL 2=缺文件）

## 缓存坑与重截教训（2026-10-15 全宽布局改造时踩）

- 重建 `dist/` 后，浏览器本地缓存（旧 `index.html` 启发式缓存 + 旧哈希 JS 整页缓存）
  会让 `open` 同一 URL 仍跑旧 bundle —— 页面能渲染不代表是新构建。
  - 验证方法：eval `Array.from(document.scripts).map(s=>s.src.split('/').pop())` 看实际 bundle 哈希
  - 临时绕过：开页 URL 加 cache-bust query（如 `?cb=<日期>`）
  - 永久修复：`server/static.py` 已给 `index.html` 加 `Cache-Control: no-cache`、
    哈希资产 `public, max-age=31536000, immutable`；干净 profile 下裸 URL 直接拿新 bundle
- 每个新浏览器会话都要重新 `set viewport 1920 1080`（默认会话窗口是 1280）
- `agent_browser_code` 单格 browser 调用上限 25 次 → 8 张截图拆两格（每格 4 张）
- 2026-10-15 基线更新：五视图根容器 `max-w-4xl` → `w-full`（右侧留白消除，卡片右缘 x≈1911/1920）

## 环境状态记录（2026-07-14 基线建立时）

- GPU 就绪（绿色徽标）；显存/利用率显示 `—`（GPU 被 vllm 占用，pynvml 读不到时回落）
- 顶栏 `ModelOpt: unknown`（repo 未检出 `~/Model-Optimizer`，见设置区探测明细 `repo_ok=✗`）
- 量化区默认 qformat=nvfp4、KV=none（不量化 KV）；数据集列表来自后端能力探测
- 空态文案：配方区「还没有 quant job」、仓库区「暂无产物 (task_dir 下没有带 run.log 的目录)」、任务区「暂无任务」
- 控制台：五区往返切换后 `errors` 为空

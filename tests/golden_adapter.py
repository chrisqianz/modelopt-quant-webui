"""把「当前代码怎么拼命令」包一层，重构时只改这里，fixture 不动。

Task 5 拔掉 host 维度后，命令里的路径来自模块级 `app.CFG.local`，所以这里
在 import 之后把 `app.CFG` 换成一份**写死绝对路径**的配置：golden 里出现的是
`/home/tester/...`，跟跑测试这台机器的真实路径无关。

三处以磁盘上的真实代码为准：

1. `app.build_cmd(p)` 收的是 `build_params` 之后的**字典**（内部走
   `p.get("model")`），不是 gradio 的位置参数列表。所以 assemble_quantize 走 UI
   的真实链路：位置参数 → `app.build_params` → `app.build_cmd`。
2. `app.INPUT_ORDER` 里没有 host_name（host 维度已删），`build_params` 的
   `a[0]` 就是 `model`。
3. 路径全部给绝对路径，避免 `resolve_relative` 把相对路径挂到项目根上 ——
   那样 golden 会烘进「这台机器的 checkout 在哪」。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  import 无副作用：main() 在 __main__ 守卫里
from webui.config import AppConfig, LocalConfig, NetConfig  # noqa: E402

# 钉死执行侧路径：不读 config.yaml / net.json，不碰真实 HOME。
# 与 Task 5 之前「本机模式 + 默认路径 + HOME=/home/tester」展开出来的结果一致，
# 所以 golden 不需要重生成 —— 命令串没变就是行为没变。
app.CFG = AppConfig(
    local=LocalConfig(
        repo="/home/tester/Model-Optimizer",
        venv="/home/tester/.venvs/modelopt",
        task_dir="/home/tester/quant-webui-tasks",
        hf_cache="/home/tester/.cache/huggingface",
    ),
    net=NetConfig(),
)


def quantize_args(c: dict) -> list:
    """用例 dict → gradio 事件的位置参数列表（顺序照 app.INPUT_ORDER）。"""
    names = c["sizes_names"]
    table = [[c["sizes"][n]] for n in names]
    row = {**c, "sizes_table": table}
    return [row[k] for k in app.INPUT_ORDER if k in row]


def assemble_quantize(a: list) -> str:
    """gradio 位置参数 → 要执行的命令串（UI「启动量化」走过的同一条链路）。"""
    return app.build_cmd(app.build_params(a))


def assemble_eval(quant: str, baseline: str, tasks: list[str],
                  tp: int, port: int, batch: int | str) -> str:
    # build_eval_script 内部就是 str(batch or "auto")，先转字符串不改变行为
    return app.build_eval_script(quant, baseline, tasks, tp, port, str(batch))


def assemble_tables(results: list[dict]) -> list[list[str]]:
    rows, _note = app._result_tables(results)
    return rows

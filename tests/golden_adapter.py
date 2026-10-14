"""把「当前代码怎么拼命令」包一层，重构时只改这里，fixture 不动。

Task 5 拔掉 host 维度时，只改 assemble_quantize / assemble_eval 的实现，
golden/*.json 必须原样通过 —— 这是「行为没变」的唯一证据。

两处按计划正文写会跑不起来，这里以磁盘上的真实代码为准：

1. `app.build_remote_cmd(host, p)` 收的是 `build_params` 之后的**字典**（内部走
   `p.get("model")`），不是 gradio 的位置参数列表。所以 assemble_quantize 走 UI
   的真实链路：位置参数 → `app.build_params` → `app.build_remote_cmd`。

2. host 不取 `app.CFG.get_host(app.CFG.host_names()[0])`：那台机器在 gitignore 的
   `hosts.json` 里（换机器/删文件即变），且它的 mode 缺省被推断成 remote →
   `q_remote` 会真的 ssh 一次取 $HOME（ConnectTimeout=10）。命令行前缀会随
   「GPU 机开没开机」在 `/home/xxx/...` 与 `~/...` 之间漂，golden 就不可复现。
   这里钉一台「播种主机」形状的 HostConfig（mode=local + config 默认路径），不联网。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  import 无副作用：main() 在 __main__ 守卫里
from webui.config import (  # noqa: E402
    DEFAULT_HF_CACHE,
    DEFAULT_REPO,
    DEFAULT_TASK_DIR,
    DEFAULT_VENV,
    MODE_LOCAL,
    HostConfig,
)

# 钉死执行侧主机：本机模式 + config.yaml 默认路径，不读 hosts.json、不发 ssh。
HOST = HostConfig(
    name="golden", ssh="", mode=MODE_LOCAL,
    remote_repo=DEFAULT_REPO, venv_dir=DEFAULT_VENV,
    task_dir=DEFAULT_TASK_DIR, hf_cache=DEFAULT_HF_CACHE,
)


def quantize_args(c: dict) -> list:
    """用例 dict → gradio 事件的位置参数列表（a[0]=host_name，其余按 INPUT_ORDER）。"""
    names = c["sizes_names"]
    table = [[c["sizes"][n]] for n in names]
    row = {**c, "sizes_table": table}
    return [HOST.name] + [row[k] for k in app.INPUT_ORDER if k in row]


def assemble_quantize(a: list) -> str:
    """gradio 位置参数 → 要执行的命令串（UI「启动量化」走过的同一条链路）。"""
    return app.build_remote_cmd(HOST, app.build_params(a))


def assemble_eval(quant: str, baseline: str, tasks: list[str],
                  tp: int, port: int, batch: int | str) -> str:
    # build_eval_script 内部就是 str(batch or "auto")，先转字符串不改变行为
    return app.build_eval_script(HOST, quant, baseline, tasks, tp, port, str(batch))


def assemble_tables(results: list[dict]) -> list[list[str]]:
    rows, _note = app._result_tables(results)
    return rows

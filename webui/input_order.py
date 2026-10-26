"""量化表单参数的固定输入顺序 + 配置工厂（单入口迁移时从 app.py 拔出来的公共件）。

存在理由：`tests/golden_adapter.py` 需要「事件绑定的参数顺序」和「一份写死绝对
路径的配置」来复现拼命令链路，但这两样都不该依赖 `app.py`（gradio UI 层，
单入口迁移后会改名/退役）。放到 `webui.input_order` 让 golden 与 UI 层解耦：
UI（app.py）和测试都从这里取同一份事实源。

改这里会同时影响 UI 的事件绑定顺序和 golden 复现链路 —— 两者必须一致，
所以顺序就是唯一事实源，别动。
"""

from __future__ import annotations

from webui.config import AppConfig, LocalConfig, NetConfig

# 参数输入区的固定顺序（事件绑定的唯一事实源，别动顺序）
INPUT_ORDER = [
    "model", "trust_remote_code", "qformat", "recipe", "kv_cache_qformat",
    "datasets", "sizes_names", "sizes_table", "custom_datasets", "calib_seq",
    "calib_with_images", "export_path", "batch_size", "device",
    "attn_implementation", "use_fsdp2", "cpu_offload", "low_memory_mode",
    "gpu_max_mem_percentage", "skip_generate", "use_seq_device_map",
]


def build_initial_config(repo: str, venv: str, task_dir: str, hf_cache: str) -> AppConfig:
    """构造一份纯 local + net 默认的 AppConfig（不读 config.yaml / net.json）。

    golden 复现链路用它把执行侧路径钉死成写死的绝对路径，避免把「这台机器的
    checkout 在哪」烘进命令串。
    """
    return AppConfig(
        local=LocalConfig(repo=repo, venv=venv, task_dir=task_dir, hf_cache=hf_cache),
        net=NetConfig(),
    )

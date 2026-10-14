"""ModelOpt 量化 WebUI。

独立、可移植的 Gradio 工具：把 examples/hf_ptq/hf_ptq.py 的 NVFP4（及其他
qformat）量化流程变成参数面板 + 本机执行器。

设计约束：
- 不在 ModelOpt 仓库内，不 import modelopt，本地无需 CUDA
- 全部可变项（路径 / 代理）外置于 config.yaml
- 数据集与 qformat 词表以静态快照兜底，运行时从本机仓库动态扫描
"""

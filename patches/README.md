# modelopt 补丁

本目录存放对上游 `~/Model-Optimizer`（NVIDIA ModelOpt）的本地补丁。
WebUI 本身不改 ModelOpt，量化行为差异通过"在量化 venv 对应的 ModelOpt 仓库上打补丁"实现。

## 应用方法

在跑量化的机器上，进入 ModelOpt 仓库根目录（含 `modelopt/` 包的那层）：

```bash
cd ~/Model-Optimizer
git apply --check ../modelopt-quant-webui/patches/modelopt-agnes-decoder-discovery.patch
git apply       ../modelopt-quant-webui/patches/modelopt-agnes-decoder-discovery.patch
```

> 用 `patch -p1 < ...` 也可以。应用前先确认该文件没被别的改动碰过，
> `git apply --check` 不通过就人工合并。

---

## modelopt-agnes-decoder-discovery.patch

**问题**：Agnes（`AgnesForConditionalGeneration` / `model_type=agnes_theta`，MoE + GDN 线性注意力）
在 **offload 场景**下导出 HF checkpoint 时强制走 streaming 路径
（`unified_export_hf.py` 的 `_offloaded` 分支），而 streaming 导出要求
`LayerActivationCollector.get_decoder_layers()` 能发现 decoder 层。

通用的 homogeneous discoverer 只沿 `model → .model → .language_model → .layers` 这条链找，
而 Agnes 的 decoder 层在 `model.core.text_engine.stages`（`nn.ModuleList[DecoderStage]`），
这条链到不了 → `get_decoder_layers` 返回 `None` → 抛

```text
RuntimeError: Streaming export requires discoverable decoder layers.
The model architecture is not supported by LayerActivationCollector.
```

**修法**：为 Agnes 注册一个专属 discoverer，直接读 `.core.text_engine.stages`。
`nn.ModuleList` 类型判断保证不会误伤其它架构（负例普通 `nn.Linear` 返回 `None`）。

**为什么必须在量化侧改**：这个发现逻辑属于 ModelOpt 的 streaming exporter，
WebUI 只负责拼命令、启动 hf_ptq，改不到内部结构。补丁随部署在目标机上应用一次即可。

**验证**（某台 32GB 显存的 GPU 机，Agnes 家族 ~350GB BF16 多模态模型）：
- 应用前：校准全过，导出阶段崩（同环境连续两次失败）
- 应用后：作业 `done`，流式导出零 Traceback，产物 170G（70G NVFP4 权重 + ~95G 不可量化的
  lexical lookup_table BF16 静态表 + 其余小表），dtype 账目与 4.5-bit 理论完全闭环。

冒烟测试（不载真实模型，用 fake 同构模块）：Agnes 结构 → 发现 2 层；普通 `nn.Linear` → `None`。

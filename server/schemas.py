"""pydantic v2 请求/响应模型层（与 webui.cmds.build_params 的 key 对齐）。"""
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "QuantizeParams",
    "PreviewRequest",
    "EvalStart",
    "RecipeDiffRequest",
    "PathsRequest",
    "NetRequest",
    "ErrorResponse",
]


class QuantizeParams(BaseModel):
    """量化参数。字段名与 build_params 返回 dict 的 key 一一对应。

    默认值是「字段缺省时」的兜底；真实默认以 load_config().defaults
    （build_params(a, cfg) 用的 cfg.defaults）为准。
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    trust_remote_code: bool = False
    qformat: str
    recipe: str = ""
    kv_cache_qformat: str = "none"
    datasets: list[str]
    dataset_sizes: dict[str, int] = {}
    custom_datasets: list[list[str]] = []
    calib_seq: int = 512
    calib_with_images: bool = False
    export_path: str = ""
    batch_size: int = 1
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    use_fsdp2: bool = False
    cpu_offload: bool = False
    low_memory_mode: bool = False
    gpu_max_mem_percentage: float = 0.0
    skip_generate: bool = False
    extra_args: str = ""  # 附加参数逃生口（本计划只存不拼）

    def to_build_params_dict(self) -> dict:
        """字段名已对齐 build_params 的 key，直接 dump。"""
        return self.model_dump()


class PreviewRequest(QuantizeParams):
    """Preview（试跑）请求，字段与量化参数一致。"""


class EvalStart(BaseModel):
    """起评测（vLLM serve + lm-eval）的请求。"""

    model_config = ConfigDict(extra="forbid")

    quant: str
    baseline: str = ""
    tp: int = 1
    port: int = Field(default=8000, ge=1, le=65535)
    batch: str = "auto"
    tasks: list[str] = []


class RecipeDiffRequest(BaseModel):
    """对比本次量化与官方 recipe 结果的请求。"""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    official: str = ""


class PathsRequest(BaseModel):
    """四个本机路径（本地配置区）。"""

    model_config = ConfigDict(extra="forbid")

    repo: str
    venv: str
    task_dir: str
    hf_cache: str


class NetRequest(BaseModel):
    """网络模式：auto（先直连后代理）/ manual / direct。"""

    model_config = ConfigDict(extra="forbid")

    mode: str
    proxy: str = ""


class ErrorResponse(BaseModel):
    """统一错误响应：{code, message, detail}。"""

    model_config = ConfigDict(extra="forbid")

    error: dict

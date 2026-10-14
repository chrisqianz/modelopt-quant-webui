"""pydantic 模型层的最小契约测试（Task 2）。"""
from pydantic import ValidationError

import pytest

from server.schemas import (
    EvalStart,
    ErrorResponse,
    NetRequest,
    PathsRequest,
    QuantizeParams,
)


def test_quantize_params_dumps_all_20_keys() -> None:
    q = QuantizeParams(
        model="x",
        qformat="nvfp4",
        datasets=["cnn_dailymail"],
        dataset_sizes={"cnn_dailymail": 1024},
    )
    d = q.to_build_params_dict()
    # 核心契约：字段名与 build_params 的 19 个 key 一一对应，再 + extra_args
    assert set(d) == {
        "model", "trust_remote_code", "qformat", "recipe", "kv_cache_qformat",
        "datasets", "dataset_sizes", "custom_datasets", "calib_seq", "calib_with_images",
        "export_path", "batch_size", "device", "attn_implementation", "use_fsdp2",
        "cpu_offload", "low_memory_mode", "gpu_max_mem_percentage", "skip_generate",
    } | {"extra_args"}
    assert d["model"] == "x"
    assert d["qformat"] == "nvfp4"
    assert d["datasets"] == ["cnn_dailymail"]
    assert d["dataset_sizes"] == {"cnn_dailymail": 1024}
    assert d["extra_args"] == ""


def test_quantize_params_custom_datasets_mixed_types() -> None:
    """回归：gradio 表格的条数列给的是 int（如 [['mysite', 128]]），schema 不能拒绝。"""
    q = QuantizeParams(
        model="x", qformat="nvfp4", datasets=[],
        custom_datasets=[["mysite", 128]],
    )
    assert q.custom_datasets == [["mysite", 128]]
    # 名称列允许纯字符串行（单列表格折一维后的形状由前端统一）
    q2 = QuantizeParams(model="x", qformat="nvfp4", datasets=[],
                        custom_datasets=[["tiny"]])
    assert q2.custom_datasets == [["tiny"]]


def test_error_response() -> None:
    e = ErrorResponse(error={"code": "x", "message": "y", "detail": "z"})
    assert e.error == {"code": "x", "message": "y", "detail": "z"}


def test_eval_start_port_range() -> None:
    assert EvalStart(quant="q").port == 8000
    for bad in (0, 70000, -1):
        with pytest.raises(ValidationError):
            EvalStart(quant="q", port=bad)


def test_paths_request_and_net_request() -> None:
    p = PathsRequest(repo="a", venv="b", task_dir="c", hf_cache="d")
    assert p.repo == "a" and p.venv == "b" and p.task_dir == "c" and p.hf_cache == "d"
    n = NetRequest(mode="auto")
    assert n.mode == "auto" and n.proxy == ""


def test_quantize_params_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        QuantizeParams.model_validate(
            {
                "model": "x",
                "qformat": "nvfp4",
                "datasets": [],
                "dataset_sizes": {},
                "unknown_field": 1,
            }
        )

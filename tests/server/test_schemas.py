"""pydantic 模型层的最小契约测试（Task 2）。"""
import pytest
from pydantic import ValidationError

from server.schemas import (
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
    assert len(d) == 20  # build_params 的 19 个 key + extra_args
    assert d["model"] == "x"
    assert d["qformat"] == "nvfp4"
    assert d["datasets"] == ["cnn_dailymail"]
    assert d["dataset_sizes"] == {"cnn_dailymail": 1024}
    assert d["extra_args"] == ""


def test_error_response() -> None:
    e = ErrorResponse(error={"code": "x", "message": "y", "detail": "z"})
    assert e.error == {"code": "x", "message": "y", "detail": "z"}


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

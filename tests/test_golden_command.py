import json
from pathlib import Path

import pytest

from tests import golden_adapter as ad
from tests.cases import EVAL_CASES, QUANTIZE_CASES
from webui.cmds import _validate_extra_args

GOLDEN = Path(__file__).parent / "golden"
FIX = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((GOLDEN / f"{name}.json").read_text())


def _dump(name: str, obj) -> None:
    GOLDEN.mkdir(exist_ok=True)
    (GOLDEN / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture
def update(request):
    return request.config.getoption("--update-golden")


def test_quantize_cmd(update):
    want = {k: ad.assemble_quantize(ad.quantize_args(v),
                                    extra_args=v.get("extra_args", ""))
            for k, v in QUANTIZE_CASES.items()}
    if update:
        _dump("quantize_cmd", want)
    assert want == _load("quantize_cmd")


def test_eval_script(update):
    want = {k: ad.assemble_eval(**v) for k, v in EVAL_CASES.items()}
    if update:
        _dump("eval_script", want)
    assert want == _load("eval_script")


def test_result_tables(update):
    sample = json.loads((FIX / "lm_eval_sample.json").read_text())
    want = ad.assemble_tables([{"file": "quant", "data": sample},
                              {"file": "baseline", "data": sample}])
    if update:
        _dump("result_tables", want)
    assert want == _load("result_tables")


def test_extra_args_forbidden_chars():
    """逃生口禁命令操作符：; & | < > $ ` 换行 一律拒。"""
    for bad in ("a;b", "--x 1 & --y 2", "echo hi | tee", "rm < f", "x $y", "`cmd`", "a\nb"):
        with pytest.raises(ValueError):
            _validate_extra_args(bad)


def test_extra_args_unbalanced_quote_rejected():
    """引号不平衡（shlex 解析失败）也要拒。"""
    with pytest.raises(ValueError):
        _validate_extra_args("--x 'unbalanced")


def test_extra_args_ok_shapes():
    """合法形状：flag、flag=value、flag 带值、路径值（含 / : .）。"""
    for ok in ("--verbose", "--max_gpu_memory_gb 24", "--offload_folder /tmp/a:b.c",
               "--specdec_offline_dataset ds --inference_tensor_parallel 2"):
        _validate_extra_args(ok)  # 不抛即通过

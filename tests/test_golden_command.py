import json
from pathlib import Path

import pytest

from tests import golden_adapter as ad
from tests.cases import EVAL_CASES, QUANTIZE_CASES

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
    want = {k: ad.assemble_quantize(ad.quantize_args(v)) for k, v in QUANTIZE_CASES.items()}
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

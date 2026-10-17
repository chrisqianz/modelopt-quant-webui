"""recipe 菜单（webui.vocab.scan_recipes / check_recipe）。

不依赖真仓库：tmp 里造 modelopt_recipes/ 树，钉死两件事——
判据（顶层 quantize 段才算完整 recipe）与解析顺序（先相对仓库、再
modelopt_recipes/ 下，探测 原样/.yaml/.yml；绝对路径直探）。
"""
import pytest

from webui.vocab import check_recipe, scan_recipes

_OK = """\
imports:
  base_disable_all: configs/ptq/units/base_disable_all
metadata:
  description: >-
    NVFP4 experts only, FP8 KV cast.
quantize:
  algorithm:
    method: max
"""
_COMPONENT = "cfg:\n  num_bits: [2, 1]\n"          # 无 quantize 段 → 组件，不进菜单
_BROKEN = "quantize:\n\tbad_indent: 1\n"            # yaml 解析炸 → 进 skipped


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo" / "modelopt_recipes"
    (root / "general" / "ptq").mkdir(parents=True)
    (root / "general" / "ptq" / "nvfp4_default-kv_fp8_cast.yaml").write_text(_OK)
    (root / "general" / "ptq" / "_broken.yml").write_text(_BROKEN)
    (root / "configs" / "numerics").mkdir(parents=True)
    (root / "configs" / "numerics" / "nvfp4.yml").write_text(_OK)   # 有 quantize 但故意不扫
    (root / "models" / "Qwen" / "QwQ" / "ptq").mkdir(parents=True)
    (root / "models" / "Qwen" / "QwQ" / "ptq" / "a_mse.yaml").write_text(_OK)
    return tmp_path / "repo"


def test_scan_menu_judgment(repo):
    out = scan_recipes(repo)
    assert out["error"] == ""
    rows = out["recipes"]
    assert isinstance(rows, list)
    got = {r["name"]: r for r in rows}
    assert set(got) == {"general/ptq/nvfp4_default-kv_fp8_cast",
                        "models/Qwen/QwQ/ptq/a_mse"}      # configs/ 不列；坏文件不进表
    skipped = out["skipped"]
    assert isinstance(skipped, list) and skipped and "_broken.yml" in skipped[0]
    assert got["general/ptq/nvfp4_default-kv_fp8_cast"]["group"] == "官方通用"
    assert got["models/Qwen/QwQ/ptq/a_mse"]["group"] == "模型专属"
    assert got["general/ptq/nvfp4_default-kv_fp8_cast"]["description"].startswith("NVFP4 experts")


def test_scan_degrades_without_repo(tmp_path):
    out = scan_recipes(tmp_path / "nope")
    assert out["recipes"] == [] and out["error"]           # 前端据此退回纯文本输入


def test_check_recipe_resolution(repo):
    assert check_recipe(repo, "") is None                          # 空=不传 --recipe
    assert check_recipe(repo, "general/ptq/nvfp4_default-kv_fp8_cast") is None  # 省后缀
    assert check_recipe(repo, "general/ptq/_broken") is None       # 存在但解析炸 → 仍算存在（loader 的错留给它报）
    assert check_recipe(repo, "models/Qwen/QwQ/ptq/a_mse.yaml") is None         # 带后缀也行
    abs_ok = repo / "my" / "custom.yaml"
    abs_ok.parent.mkdir(parents=True)
    abs_ok.write_text(_OK)
    assert check_recipe(repo, str(abs_ok)) is None                 # 绝对路径直探
    assert check_recipe(repo, "general/ptq/nope") is not None      # 不存在的名字 → 拦
    msg = check_recipe(repo, "general/ptq/nope")
    assert msg is not None and "从菜单里选" in msg
    assert check_recipe(repo, str(repo / "nope.yaml")) is not None  # 不存在的绝对路径 → 拦


def test_check_recipe_skips_when_repo_unreadable(tmp_path):
    # 与 capabilities 探测失败不拦截同理：仓库不可读宁可漏拦不误杀
    assert check_recipe(tmp_path / "nope", "whatever") is None

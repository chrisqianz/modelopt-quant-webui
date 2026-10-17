"""官方对照：本地路径支持 + curl 退出码措辞（对照用户报的「HTTP 5」误导）。"""
import json

import pytest

from webui import validate


def _qc_config():
    return {"quantization_config": {
        "quant_method": "modelopt",
        "config_groups": {"group_0": {"weights": {"num_bits": [2, 1]},
                                      "targets": ["*mlp*"]}},
        "ignore": ["*visual*"]}}


def test_local_dir_path_reads_config_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qc_config()))
    s = validate.fetch_official_summary(str(tmp_path))
    assert s.error == ""
    assert s.groups and s.groups[0]["bits"] == [2, 1]
    assert s.ignore == ["*visual*"]


def test_local_config_json_file_path(tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps(_qc_config()))
    s = validate.fetch_official_summary(str(f))
    assert s.groups and s.groups[0]["count"] == 1


def test_local_path_missing_raises_friendly(tmp_path):
    with pytest.raises(RuntimeError) as e:
        validate.fetch_official_summary(str(tmp_path / "nope"))
    assert "本地对照文件不存在" in str(e.value)
    assert "owner/name" in str(e.value)   # 提示：想用 HF 卡该填什么


def test_local_bf16_model_explains_no_quant_config(tmp_path):
    """原始 BF16 模型没有 quantization_config —— 要说清「不是 bug，是没得对」。"""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["X"]}))
    with pytest.raises(RuntimeError) as e:
        validate.fetch_official_summary(str(tmp_path))
    assert "quantization_config" in str(e.value)
    assert "原始模型" in str(e.value)


def test_curl_exit_code_is_not_reported_as_http(monkeypatch):
    """curl 退出码 5（连不上代理）不能被写成「HTTP 5」——那是用户踩过的误导。"""
    class P:
        returncode, stdout, stderr = 5, "", ""
    monkeypatch.setattr(validate.subprocess, "run", lambda *a, **k: P())
    with pytest.raises(RuntimeError) as e:
        validate.fetch_hf_quant_config("nvidia/Fake-NVFP4")
    msg = str(e.value)
    assert "curl exit 5" in msg and "HTTP 5" not in msg
    assert "连不上代理" in msg


def test_http_status_error_when_curl_ok_but_404(monkeypatch):
    class P:
        returncode, stdout, stderr = 0, '{"error":"not found"}\n@@HTTP:404', ""
    monkeypatch.setattr(validate.subprocess, "run", lambda *a, **k: P())
    with pytest.raises(RuntimeError) as e:
        validate.fetch_hf_quant_config("nvidia/Missing")
    assert "HTTP 404" in str(e.value)


def test_hf_base_model_card_says_nothing_to_compare(monkeypatch):
    """HF 原模型卡（有 config 无 quantization_config）也要说人话，
    不能退化成「无法对照：缺少 config_groups」。"""
    class P:
        returncode = 0
        stdout = '{"config":{"architectures":["Qwen3ForCausalLM"]}}\n@@HTTP:200'
        stderr = ""
    monkeypatch.setattr(validate.subprocess, "run", lambda *a, **k: P())
    with pytest.raises(RuntimeError) as e:
        validate.fetch_official_summary("Qwen/Qwen3.5-4B")
    msg = str(e.value)
    assert "原始模型" in msg and "量化模型卡" in msg

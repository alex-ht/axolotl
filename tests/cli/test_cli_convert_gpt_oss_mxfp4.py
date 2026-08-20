"""Tests for axolotl convert-gpt-oss-mxfp4."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from axolotl.cli.convert_gpt_oss_mxfp4 import (
    _MODELOPT_INSTALL,
    _require_mxfp4_qtensor,
    do_cli,
    to_oai_mxfp4_weight_only,
    write_oai_mxfp4_config,
)
from axolotl.cli.main import cli


class _FakeMXFP4QTensor:
    def __init__(self, quantized_data):
        self._quantized_data = quantized_data

    @classmethod
    def quantize(cls, expert, block_size=32):
        rows, cols = expert.shape[-2], expert.shape[-1]
        packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device=expert.device)
        scales = torch.ones(
            rows, cols // block_size, dtype=torch.uint8, device=expert.device
        )
        return cls(packed), scales


class _FakeModel:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def test_help_lists_command(cli_runner):
    result = cli_runner.invoke(cli, ["convert-gpt-oss-mxfp4", "--help"])
    assert result.exit_code == 0
    assert "--model-path" in result.output
    assert "--lora-path" in result.output
    assert "--output-path" in result.output


def test_output_path_required(cli_runner):
    result = cli_runner.invoke(cli, ["convert-gpt-oss-mxfp4", "--model-path", "foo"])
    assert result.exit_code != 0
    assert "output-path" in result.output


def test_cli_dispatches_to_do_cli(cli_runner):
    with patch("axolotl.cli.convert_gpt_oss_mxfp4.do_cli") as mock_do_cli:
        result = cli_runner.invoke(
            cli,
            [
                "convert-gpt-oss-mxfp4",
                "--model-path",
                "ckpt",
                "--output-path",
                "out",
            ],
        )
        assert result.exit_code == 0
        mock_do_cli.assert_called_once_with(
            output_path="out",
            model_path="ckpt",
            lora_path=None,
            base_path=None,
            trust_remote_code=False,
        )


def test_cli_lora_flags(cli_runner):
    with patch("axolotl.cli.convert_gpt_oss_mxfp4.do_cli") as mock_do_cli:
        result = cli_runner.invoke(
            cli,
            [
                "convert-gpt-oss-mxfp4",
                "--lora-path",
                "adapter",
                "--base-path",
                "openai/gpt-oss-20b",
                "--output-path",
                "out",
                "--trust-remote-code",
            ],
        )
        assert result.exit_code == 0
        mock_do_cli.assert_called_once_with(
            output_path="out",
            model_path=None,
            lora_path="adapter",
            base_path="openai/gpt-oss-20b",
            trust_remote_code=True,
        )


def test_model_path_and_lora_path_mutually_exclusive():
    with pytest.raises(ValueError, match="only one"):
        do_cli(output_path="out", model_path="ckpt", lora_path="adapter")


def test_lora_requires_base_path():
    with pytest.raises(ValueError, match="--base-path"):
        do_cli(output_path="out", lora_path="adapter")


def test_missing_source_path():
    with pytest.raises(ValueError, match="--model-path"):
        do_cli(output_path="out")


def test_missing_modelopt_message(monkeypatch):
    monkeypatch.setattr(
        "axolotl.cli.convert_gpt_oss_mxfp4._require_mxfp4_qtensor",
        lambda: (_ for _ in ()).throw(ImportError(_MODELOPT_INSTALL)),
    )
    with pytest.raises(ImportError, match=r"axolotl\[modelopt\]"):
        do_cli(output_path="out", model_path="ckpt")


def test_require_mxfp4_qtensor_wraps_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "modelopt" or name.startswith("modelopt."):
            raise ImportError("No module named modelopt")
        return real_import(name, *args, **kwargs)

    for key in list(sys.modules):
        if key == "modelopt" or key.startswith("modelopt."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"axolotl\[modelopt\]"):
        _require_mxfp4_qtensor()


def test_to_oai_mxfp4_weight_only_converts_experts_only():
    state = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 32, 64),
        "model.layers.0.mlp.experts.gate_up_proj_bias": torch.randn(2, 64),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
    }
    out = to_oai_mxfp4_weight_only(_FakeModel(state), mxfp4_qtensor=_FakeMXFP4QTensor)

    assert "model.layers.0.mlp.experts.gate_up_proj" not in out
    blocks = out["model.layers.0.mlp.experts.gate_up_proj_blocks"]
    scales = out["model.layers.0.mlp.experts.gate_up_proj_scales"]
    assert blocks.shape == (2, 64, 1, 16)
    assert scales.shape == (2, 64, 1)
    assert torch.equal(
        out["model.layers.0.mlp.experts.gate_up_proj_bias"],
        state["model.layers.0.mlp.experts.gate_up_proj_bias"],
    )
    assert torch.equal(
        out["model.layers.0.self_attn.q_proj.weight"],
        state["model.layers.0.self_attn.q_proj.weight"],
    )


def test_write_oai_mxfp4_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model_type": "gpt_oss", "torch_dtype": "bfloat16"}')

    write_oai_mxfp4_config(str(tmp_path))

    parsed = json.loads((tmp_path / "config.json").read_text())
    assert "torch_dtype" not in parsed
    assert parsed["quantization_config"]["quant_method"] == "mxfp4"
    assert (
        "model.layers.*.self_attn"
        in parsed["quantization_config"]["modules_to_not_convert"]
    )


def test_do_cli_convert_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "axolotl.cli.convert_gpt_oss_mxfp4._require_mxfp4_qtensor",
        lambda: _FakeMXFP4QTensor,
    )

    state = {
        "model.layers.0.mlp.experts.down_proj": torch.randn(2, 32, 64),
        "lm_head.weight": torch.randn(8, 32),
    }
    fake_model = MagicMock()
    fake_model.state_dict.return_value = state
    fake_tokenizer = SimpleNamespace()
    fake_tokenizer.save_pretrained = MagicMock()

    def fake_save(output_path, state_dict=None):
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        (out / "config.json").write_text('{"model_type": "gpt_oss"}')
        fake_model.saved_state = state_dict

    fake_model.save_pretrained.side_effect = fake_save

    monkeypatch.setattr(
        "axolotl.cli.convert_gpt_oss_mxfp4._load_model",
        lambda **_kwargs: (fake_model, fake_tokenizer),
    )

    out_dir = str(tmp_path / "out")
    do_cli(output_path=out_dir, model_path="ckpt")

    fake_model.save_pretrained.assert_called_once()
    saved = fake_model.saved_state
    assert "model.layers.0.mlp.experts.down_proj_blocks" in saved
    assert "lm_head.weight" in saved
    fake_tokenizer.save_pretrained.assert_called_once_with(out_dir)
    parsed = json.loads((tmp_path / "out" / "config.json").read_text())
    assert parsed["quantization_config"]["quant_method"] == "mxfp4"

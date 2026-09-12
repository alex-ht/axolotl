"""Tests for fused-linear EAFT ForCausalLM.forward patching."""

import pytest
import torch

from axolotl.monkeypatch.loss.eaft_patch import (
    _eaft_from_lm_head,
    make_eaft_forward,
    patch_eaft_forward,
)


class _NoBiasWrapper:
    """Mirrors PEFT ``TrainableTokensWrapper``: ``.weight`` exists, ``.bias`` raises."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def __getattr__(self, name: str):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def test_patch_eaft_forward_resolves_gemma3_text_module():
    """gemma3_text lives in transformers.models.gemma3, not gemma3_text."""
    gemma3_modeling = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
    model_cls = gemma3_modeling.Gemma3ForCausalLM
    original_forward = model_cls.forward
    try:
        patch_eaft_forward("gemma3_text", alpha=1.0, k=8)
        assert model_cls.forward is not original_forward
        assert model_cls.forward.__code__ is make_eaft_forward(1.0, 8).__code__
    finally:
        model_cls.forward = original_forward


def test_eaft_from_lm_head_without_bias_attr():
    torch.manual_seed(0)
    hidden = torch.randn(1, 4, 8)
    labels = torch.tensor([[-100, 1, 2, 3]])
    weight = torch.randn(16, 8)
    loss = _eaft_from_lm_head(
        _NoBiasWrapper(weight),
        hidden,
        labels,
        shift_labels=None,
        alpha=1.0,
        k=8,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_eaft_from_peft_trainable_tokens_wrapper():
    pytest.importorskip("peft")
    from peft.utils.other import TrainableTokensWrapper

    torch.manual_seed(0)
    linear = torch.nn.Linear(8, 16, bias=False)
    lm_head = TrainableTokensWrapper(linear, "default", token_indices=[1, 2, 3])
    with pytest.raises(AttributeError, match="bias"):
        _ = lm_head.bias

    hidden = torch.randn(1, 4, 8)
    labels = torch.tensor([[-100, 1, 2, 3]])
    loss = _eaft_from_lm_head(
        lm_head,
        hidden,
        labels,
        shift_labels=None,
        alpha=1.0,
        k=8,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

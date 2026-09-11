"""Tests for fused-linear EAFT ForCausalLM.forward patching."""

import pytest

from axolotl.monkeypatch.loss.eaft_patch import make_eaft_forward, patch_eaft_forward


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

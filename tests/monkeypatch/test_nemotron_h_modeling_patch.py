"""Unit tests for Nemotron-H packing / router patches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from axolotl.monkeypatch.models.nemotron_h.modeling import (
    ATTENTION_BLOCK_TYPES,
    MAMBA_BLOCK_TYPES,
    patch_nemotron_h_router_bias_device,
    patched_nemotron_h_block_forward,
    patched_nemotron_h_router_forward,
)


class _Norm:
    def __init__(self, hidden_size: int):
        self.weight = torch.ones(hidden_size)

    def __call__(self, hidden_states):
        return hidden_states


class _FakeCache:
    def __init__(self, has_previous_state: bool = False):
        self.has_previous_state = has_previous_state


def _block(block_type: str, mixer):
    return SimpleNamespace(
        block_type=block_type,
        mixer=mixer,
        norm=_Norm(4),
    )


class TestNemotronHBlockTypeAliases:
    def test_transformer_5_14_names_are_recognized(self):
        assert "linear_attention" in MAMBA_BLOCK_TYPES
        assert "full_attention" in ATTENTION_BLOCK_TYPES

    def test_pre_5_14_aliases_are_recognized(self):
        assert "mamba" in MAMBA_BLOCK_TYPES
        assert "attention" in ATTENTION_BLOCK_TYPES

    @pytest.mark.parametrize("block_type", ["linear_attention", "mamba"])
    def test_mamba_block_threads_seq_idx(self, block_type):
        mixer = MagicMock(return_value=torch.ones(1, 3, 4))
        block = _block(block_type, mixer)
        hidden = torch.zeros(1, 3, 4)
        position_ids = torch.tensor([[0, 1, 0]])

        out = patched_nemotron_h_block_forward(block, hidden, position_ids=position_ids)

        mixer.assert_called_once()
        kwargs = mixer.call_args.kwargs
        assert kwargs["seq_idx"] is not None
        assert torch.equal(
            kwargs["seq_idx"], torch.tensor([[0, 0, 1]], dtype=torch.int32)
        )
        assert out.shape == hidden.shape

    @pytest.mark.parametrize("block_type", ["full_attention", "attention"])
    def test_attention_block_unpacks_tuple(self, block_type):
        attn_out = torch.ones(1, 2, 4)
        mixer = MagicMock(return_value=(attn_out, None))
        block = _block(block_type, mixer)
        hidden = torch.zeros(1, 2, 4)

        out = patched_nemotron_h_block_forward(block, hidden, use_cache=True)

        mixer.assert_called_once()
        assert mixer.call_args.kwargs["use_cache"] is True
        assert "user_cache" not in mixer.call_args.kwargs
        assert torch.equal(out, attn_out)

    def test_attention_tuple_without_patch_would_break_residual(self):
        mixer = MagicMock(return_value=(torch.ones(1, 2, 4), None))
        block = _block("full_attention", mixer)
        hidden = torch.zeros(1, 2, 4)

        out = patched_nemotron_h_block_forward(block, hidden)
        assert isinstance(out, torch.Tensor)

    def test_moe_block_calls_mixer_with_hidden_only(self):
        mixer = MagicMock(return_value=torch.ones(1, 2, 4))
        block = _block("moe", mixer)
        hidden = torch.zeros(1, 2, 4)

        patched_nemotron_h_block_forward(block, hidden)

        mixer.assert_called_once_with(block.norm(hidden))

    def test_mixer_non_tensor_raises_typeerror(self):
        mixer = MagicMock(return_value=(torch.ones(1, 2, 4), None))
        block = _block("moe", mixer)

        with pytest.raises(TypeError, match="returned tuple"):
            patched_nemotron_h_block_forward(block, torch.zeros(1, 2, 4))

    def test_mamba_skips_seq_idx_during_decode(self):
        mixer = MagicMock(return_value=torch.ones(1, 1, 4))
        block = _block("linear_attention", mixer)

        patched_nemotron_h_block_forward(
            block,
            torch.zeros(1, 1, 4),
            past_key_values=_FakeCache(has_previous_state=True),
            position_ids=torch.tensor([[4]]),
        )

        assert mixer.call_args.kwargs["seq_idx"] is None


class _DummyRouter(nn.Module):
    def __init__(self, num_experts=4, hidden_dim=8, num_group=2, top_k=2, topk_group=1):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.routed_scaling_factor = 1.0
        self.num_group = num_group
        self.topk_group = topk_group
        self.norm_topk_prob = True
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))


class TestNemotronHRouterBiasDevice:
    def test_patch_installs_and_is_idempotent(self):
        class Router:
            def forward(self, hidden_states):
                return hidden_states

        mod = SimpleNamespace(NemotronHTopkRouter=Router)
        original = Router.forward

        patch_nemotron_h_router_bias_device(mod)
        assert Router.forward is patched_nemotron_h_router_forward
        assert Router.forward._axolotl_offload_bias_fix is True

        patch_nemotron_h_router_bias_device(mod)
        assert Router.forward is patched_nemotron_h_router_forward
        assert original is not Router.forward

    def test_missing_router_class_is_a_noop(self):
        mod = SimpleNamespace()
        patch_nemotron_h_router_bias_device(mod)

    def test_forward_does_not_reassign_module_bias(self):
        router = _DummyRouter()
        bias_id = id(router.e_score_correction_bias)
        hidden = torch.randn(3, 8)

        logits, weights, indices = patched_nemotron_h_router_forward(router, hidden)

        assert id(router.e_score_correction_bias) == bias_id
        assert logits.shape == (3, router.num_experts)
        assert weights.shape == (3, router.top_k)
        assert indices.shape == (3, router.top_k)

    def test_forward_matches_cpu_reference(self):
        router = _DummyRouter()
        nn.init.normal_(router.weight, std=0.02)
        router.e_score_correction_bias.copy_(torch.linspace(-0.2, 0.2, 4))
        hidden = torch.randn(5, 8)

        logits, weights, indices = patched_nemotron_h_router_forward(router, hidden)

        hidden_flat = hidden.view(-1, router.hidden_dim)
        ref_logits = F.linear(
            hidden_flat.type(torch.float32), router.weight.type(torch.float32)
        )
        scores = ref_logits.sigmoid()
        scores_for_choice = scores + router.e_score_correction_bias.to(torch.float32)
        group_scores = (
            scores_for_choice.view(
                -1, router.num_group, router.num_experts // router.num_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=router.topk_group, dim=-1, sorted=False)[
            1
        ]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, router.num_group, router.num_experts // router.num_group)
            .reshape(-1, router.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), float("-inf")
        )
        ref_idx = torch.topk(scores_for_choice, k=router.top_k, dim=-1, sorted=False)[1]
        ref_w = scores.gather(1, ref_idx)
        ref_w = ref_w / (ref_w.sum(dim=-1, keepdim=True) + 1e-20)

        assert torch.allclose(logits, ref_logits)
        assert torch.equal(indices, ref_idx)
        assert torch.allclose(weights, ref_w)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cpu_bias_adds_on_cuda_scores(self):
        router = _DummyRouter()
        router.weight.data = router.weight.data.cuda()
        hidden = torch.randn(2, 8, device="cuda")

        logits, weights, indices = patched_nemotron_h_router_forward(router, hidden)

        assert logits.device.type == "cuda"
        assert weights.device.type == "cuda"
        assert indices.device.type == "cuda"
        assert router.e_score_correction_bias.device.type == "cpu"

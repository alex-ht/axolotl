"""CUDA numerics for fused linear EAFT vs the logits reference."""

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused EAFT"
)


def _ref_eaft_linear(
    hidden, weight, labels, alpha, k, bias=None, ignore_index=-100, normalize=True
):
    logits = F.linear(
        hidden.float(), weight.float(), None if bias is None else bias.float()
    )
    vocab = logits.size(-1)
    logits_v = logits.reshape(-1, vocab)
    labels_v = labels.reshape(-1)
    mask = labels_v != ignore_index
    k_eff = min(k, vocab)
    with torch.no_grad():
        topk = torch.topk(logits_v[mask], k=k_eff, dim=-1).values
        probs = F.softmax(topk, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        if normalize and k_eff > 1:
            entropy = entropy / math.log(k_eff)
        weights = torch.pow(entropy, alpha)
    ce = F.cross_entropy(logits_v[mask], labels_v[mask], reduction="none")
    return (ce * weights).mean()


class TestFusedLinearEAFT:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("k", [8, 20])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_loss_and_grads_match_reference(self, dtype, k, normalize):
        from axolotl.kernels.eaft import fused_linear_eaft_loss

        torch.manual_seed(0)
        n, d, v = 32, 64, 256
        hidden = torch.randn(n, d, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(v, d, device="cuda", dtype=dtype, requires_grad=True)
        labels = torch.randint(0, v, (n,), device="cuda")
        labels[0] = -100

        hidden_ref = hidden.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)

        loss = fused_linear_eaft_loss(
            hidden,
            weight,
            labels,
            alpha=1.0,
            k=k,
            ignore_index=-100,
            normalize=normalize,
        )
        loss_ref = _ref_eaft_linear(
            hidden_ref,
            weight_ref,
            labels,
            alpha=1.0,
            k=k,
            ignore_index=-100,
            normalize=normalize,
        )

        atol = 2e-2 if dtype == torch.bfloat16 else 2e-3
        rtol = 2e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(loss.float(), loss_ref.float(), atol=atol, rtol=rtol)

        loss.backward()
        loss_ref.backward()
        torch.testing.assert_close(
            hidden.grad.float(), hidden_ref.grad.float(), atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            weight.grad.float(), weight_ref.grad.float(), atol=atol, rtol=rtol
        )

    def test_bias_and_num_items(self):
        from axolotl.kernels.eaft import fused_linear_eaft_loss

        torch.manual_seed(1)
        n, d, v = 16, 64, 128
        hidden = torch.randn(
            n, d, device="cuda", dtype=torch.float32, requires_grad=True
        )
        weight = torch.randn(
            v, d, device="cuda", dtype=torch.float32, requires_grad=True
        )
        bias = torch.randn(v, device="cuda", dtype=torch.float32, requires_grad=True)
        labels = torch.randint(0, v, (n,), device="cuda")

        loss = fused_linear_eaft_loss(
            hidden, weight, labels, alpha=1.0, k=8, bias=bias, num_items_in_batch=n
        )
        loss_ref = _ref_eaft_linear(
            hidden.detach(), weight.detach(), labels, 1.0, 8, bias=bias.detach()
        )
        # mean vs sum/n is the same when all tokens are valid
        torch.testing.assert_close(loss, loss_ref, atol=2e-3, rtol=2e-3)
        loss.backward()
        assert hidden.grad is not None
        assert weight.grad is not None
        assert bias.grad is not None

    def test_all_ignored_returns_zero(self):
        from axolotl.kernels.eaft import fused_linear_eaft_loss

        hidden = torch.randn(4, 32, device="cuda", requires_grad=True)
        weight = torch.randn(64, 32, device="cuda", requires_grad=True)
        labels = torch.full((4,), -100, device="cuda")
        loss = fused_linear_eaft_loss(hidden, weight, labels, k=8)
        assert loss.detach().item() == 0.0
        loss.backward()

    def test_k_equals_vocab(self):
        from axolotl.kernels.eaft import fused_linear_eaft_loss

        torch.manual_seed(2)
        n, d, v = 8, 32, 32
        hidden = torch.randn(
            n, d, device="cuda", dtype=torch.float32, requires_grad=True
        )
        weight = torch.randn(
            v, d, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, v, (n,), device="cuda")
        loss = fused_linear_eaft_loss(hidden, weight, labels, k=32)
        loss_ref = _ref_eaft_linear(hidden.detach(), weight.detach(), labels, 1.0, 32)
        torch.testing.assert_close(loss, loss_ref, atol=2e-3, rtol=2e-3)

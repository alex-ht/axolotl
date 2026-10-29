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
    @pytest.mark.parametrize("stable_ce", [False, True])
    def test_loss_and_grads_match_reference(self, dtype, k, normalize, stable_ce):
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
            stable_ce=stable_ce,
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


def _fp64_margin_ref(hidden, weight, labels):
    """fp64 ``(z_t, lse_not, ce, g_t)`` from the exact (rounded-input) values."""
    z = hidden.double() @ weight.double().T
    idx = labels.unsqueeze(1)
    z_t = z.gather(1, idx).squeeze(1)
    lse_not = torch.logsumexp(z.scatter(1, idx, float("-inf")), dim=1)
    neg_m = lse_not - z_t
    return z_t, lse_not, F.softplus(neg_m), -torch.sigmoid(neg_m)


def _margin_sweep_inputs(dtype, n_vocab=512, n_dim=64, scales=None):
    """Rows whose hidden state is a scaled copy of the target row of ``weight``.

    Larger scales give larger margins ``m`` (negative up to > 20).
    """
    torch.manual_seed(0)
    weight = (torch.randn(n_vocab, n_dim, device="cuda") / math.sqrt(n_dim)).to(dtype)
    scales = scales or [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    labels = torch.randint(0, n_vocab, (len(scales),), device="cuda")
    w_t = weight[labels].float()
    hidden = (
        (w_t / w_t.norm(dim=1, keepdim=True))
        * torch.tensor(scales, device="cuda").unsqueeze(1)
        * math.sqrt(n_dim)
    )
    return hidden.to(dtype), weight, labels


class TestFusedLinearEAFTStableCE:
    """``eaft_stable_ce``: Softmax Collapse fix (arXiv:2610.31642)."""

    def test_paper_example(self):
        """z=[9,-9,-9], t=0: loss ~3.046e-8, grad ~[-3.046e-8, 1.523e-8, 1.523e-8]."""
        from axolotl.kernels.eaft import fused_linear_eaft_loss

        hidden = torch.tensor(
            [[9.0, -9.0, -9.0]], device="cuda", dtype=torch.float32, requires_grad=True
        )
        weight = torch.eye(3, device="cuda", dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([0], device="cuda")
        # alpha=0 -> EAFT weight 1, i.e. plain CE
        loss = fused_linear_eaft_loss(
            hidden, weight, labels, alpha=0.0, k=3, stable_ce=True
        )
        loss.backward()

        z = torch.tensor([9.0, -9.0, -9.0], dtype=torch.float64)
        ref_loss = -torch.log_softmax(z, 0)[0]
        ref_grad = torch.softmax(z, 0)
        ref_grad[0] -= 1.0
        assert abs(ref_loss.item() - 3.046e-8) < 1e-10
        torch.testing.assert_close(loss.double().cpu(), ref_loss, rtol=1e-4, atol=0.0)
        torch.testing.assert_close(
            hidden.grad.double().cpu()[0], ref_grad, rtol=1e-4, atol=0.0
        )
        # zero up to fp32 rounding of the three stored gradients
        g = hidden.grad.double()[0]
        assert abs(g.sum().item()) <= 1e-6 * abs(g[0].item())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_margin_sweep_forward_vs_fp64(self, dtype):
        from axolotl.kernels.eaft import fused_linear_eaft_fwd_stable, stable_softplus

        hidden, weight, labels = _margin_sweep_inputs(dtype)
        _, _, z_t, lse_not = fused_linear_eaft_fwd_stable(
            hidden.contiguous(), weight.contiguous(), weight.new_empty(0), labels
        )
        ce = stable_softplus(lse_not - z_t)
        ref_z_t, ref_lse_not, ref_ce, _ = _fp64_margin_ref(hidden, weight, labels)
        m = (ref_z_t - ref_lse_not).cpu()
        assert m.min() < 0 and m.max() > 20, m  # sweep spans the fp32 collapse point
        torch.testing.assert_close(z_t.double(), ref_z_t, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lse_not.double(), ref_lse_not, rtol=1e-5, atol=1e-5)
        # relative accuracy holds even when ce << ulp(z_t)
        torch.testing.assert_close(ce.double(), ref_ce, rtol=1e-4, atol=0.0)
        assert (ce >= 0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_margin_sweep_grad_sum_zero(self, dtype):
        """Per row: target grad matches -sigmoid(-m), is <= 0, and the row sums to 0."""
        from axolotl.kernels.eaft import fused_linear_eaft_loss

        hidden, weight, labels = _margin_sweep_inputs(dtype)
        _, _, _, ref_g_t = _fp64_margin_ref(hidden, weight, labels)
        for i in range(hidden.shape[0]):
            h = hidden[i : i + 1].detach().clone().requires_grad_(True)
            w = weight.detach().clone().requires_grad_(True)
            # zero fp32 bias: d_bias[j] == dL/dz_j for this single row
            b = torch.zeros(weight.shape[0], device="cuda", requires_grad=True)
            loss = fused_linear_eaft_loss(
                h, w, labels[i : i + 1], alpha=0.0, k=20, bias=b, stable_ce=True
            )
            loss.backward()
            g = b.grad.double()
            g_t = g[labels[i]]
            assert g_t.item() <= 0.0
            torch.testing.assert_close(g_t, ref_g_t[i], rtol=1e-4, atol=0.0)
            assert abs(g.sum().item()) <= 1e-4 * abs(g_t.item()) + 1e-30

    def test_flag_off_matches_legacy_ops_bitwise(self):
        """Default (stable_ce=False) == manual composition of the legacy ops."""
        from axolotl.kernels.eaft import (
            entropy_from_topk,
            fused_linear_eaft_bwd,
            fused_linear_eaft_fwd,
            fused_linear_eaft_loss,
            target_logits,
        )

        torch.manual_seed(3)
        n, d, v = 16, 64, 300
        hidden = torch.randn(n, d, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(v, d, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, v, (n,), device="cuda")

        h = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        loss = fused_linear_eaft_loss(h, w, labels, alpha=1.0, k=20)
        loss.backward()

        empty = weight.new_empty(0)
        lse, topk = fused_linear_eaft_fwd(hidden, weight, empty)
        ce = lse - target_logits(hidden, weight, labels, empty)
        weights = entropy_from_topk(topk, 20, 1.0, True)
        ref_loss = (ce * weights).mean()
        scale = weights.float() / float(n)
        d_h, d_w, _ = fused_linear_eaft_bwd(scale, hidden, weight, labels, lse, empty)

        assert torch.equal(loss, ref_loss)
        assert torch.equal(h.grad, d_h)
        assert torch.equal(w.grad, d_w)

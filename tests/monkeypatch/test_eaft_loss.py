"""CPU tests for the logits-based EAFT reference loss."""

import math

import pytest
import torch
import torch.nn.functional as F

from axolotl.monkeypatch.loss.eaft import (
    StableCE,
    eaft_entropy_weights,
    eaft_loss,
    stable_cross_entropy,
)


def test_eaft_loss_ignores_masked_tokens():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 16, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3]])

    class _Out:
        pass

    out = _Out()
    out.logits = logits
    loss = eaft_loss(out, labels, alpha=1.0, k=8)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_eaft_loss_all_ignored_is_zero():
    logits = torch.randn(1, 3, 8, requires_grad=True)
    labels = torch.full((1, 3), -100)

    class _Out:
        pass

    out = _Out()
    out.logits = logits
    loss = eaft_loss(out, labels, k=4)
    assert loss.detach().item() == 0.0


def test_eaft_entropy_weights_normalize_by_ln_k():
    uniform = torch.zeros(1, 8)
    w_norm = eaft_entropy_weights(uniform, k=8, alpha=1.0, normalize=True)
    w_raw = eaft_entropy_weights(uniform, k=8, alpha=1.0, normalize=False)
    torch.testing.assert_close(w_norm, torch.ones(1))
    torch.testing.assert_close(w_raw, torch.full((1,), math.log(8)))


class _Out:
    def __init__(self, logits):
        self.logits = logits


def test_stable_ce_gradcheck_fp64():
    torch.manual_seed(0)
    logits = torch.randn(5, 7, dtype=torch.float64)
    logits[0] *= 12.0  # one very confident / very wrong row
    logits.requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 6, 3])
    assert torch.autograd.gradcheck(StableCE.apply, (logits, labels))


def test_stable_ce_matches_cross_entropy_fp64():
    torch.manual_seed(1)
    logits = (torch.randn(8, 11, dtype=torch.float64) * 4).requires_grad_(True)
    labels = torch.randint(0, 11, (8,))
    ref = logits.detach().clone().requires_grad_(True)
    out = stable_cross_entropy(logits, labels)
    out_ref = F.cross_entropy(ref, labels, reduction="none")
    torch.testing.assert_close(out, out_ref)
    out.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(logits.grad, ref.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stable_ce_paper_example(dtype):
    """z=[9,-9,-9], t=0: fp32/bf16 F.cross_entropy collapses to 0; StableCE does not."""
    logits = torch.tensor([[9.0, -9.0, -9.0]], dtype=dtype, requires_grad=True)
    loss = stable_cross_entropy(logits, torch.tensor([0]))
    assert loss.dtype == torch.float32
    loss.sum().backward()
    torch.testing.assert_close(
        loss.double(), torch.tensor([3.046e-8], dtype=torch.float64), rtol=1e-3, atol=0
    )
    if dtype == torch.float32:
        expected = torch.tensor([[-3.046e-8, 1.523e-8, 1.523e-8]], dtype=torch.float64)
        torch.testing.assert_close(logits.grad.double(), expected, rtol=1e-3, atol=0)
        g = logits.grad.double()[0]
        # zero up to fp32 rounding of the stored gradients
        assert abs(g.sum().item()) <= 1e-6 * abs(g[0].item())
    else:
        assert logits.grad[0, 0].item() < 0.0


def test_stable_ce_fp32_margin_sweep_vs_fp64():
    """Per-row loss and gradients keep relative accuracy and sum to ~0 for m in [0, 40]."""
    margins = torch.tensor(
        [0.0, 5.0, 10.0, 16.0, 20.0, 30.0, 40.0], dtype=torch.float64
    )
    z64 = torch.full((len(margins), 50), -2.0, dtype=torch.float64)
    z64[:, 1:] += torch.linspace(-1, 1, 49, dtype=torch.float64)
    lse_not = torch.logsumexp(z64[:, 1:], dim=1)
    z64[:, 0] = lse_not + margins
    z32 = z64.float().requires_grad_(True)
    labels = torch.zeros(len(margins), dtype=torch.long)

    loss = stable_cross_entropy(z32, labels)
    loss.sum().backward()

    # fp64 reference in margin form: plain fp64 F.cross_entropy itself
    # collapses once m > ~37, so it cannot serve as ground truth here.
    z = z32.detach().double()
    ref_lse_not = torch.logsumexp(z[:, 1:], dim=1)
    neg_m = ref_lse_not - z[:, 0]
    exact = F.softplus(neg_m)
    torch.testing.assert_close(loss.double(), exact, rtol=1e-4, atol=0)
    g = z32.grad.double()
    assert (g[:, 0] < 0).all()
    torch.testing.assert_close(g[:, 0], -torch.sigmoid(neg_m), rtol=1e-4, atol=0)
    row_sum = g.sum(dim=1).abs()
    assert (row_sum <= 1e-5 * g[:, 0].abs()).all(), row_sum


def test_eaft_loss_stable_flag_off_unchanged():
    torch.manual_seed(2)
    logits = torch.randn(1, 6, 32)
    labels = torch.tensor([[-100, 1, 2, 3, 4, 5]])
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    la = eaft_loss(_Out(a), labels, alpha=1.0, k=8)
    lb = eaft_loss(_Out(b), labels, alpha=1.0, k=8, stable_ce=False)
    la.backward()
    lb.backward()
    assert torch.equal(la, lb)
    assert torch.equal(a.grad, b.grad)


def test_eaft_loss_stable_close_to_legacy_and_weights_detached():
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 32)
    labels = torch.tensor([[-100, 1, 2, 3, 4, 5], [7, 8, -100, 9, 10, 11]])
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    legacy = eaft_loss(_Out(a), labels, alpha=1.0, k=8)
    stable = eaft_loss(_Out(b), labels, alpha=1.0, k=8, stable_ce=True)
    legacy.backward()
    stable.backward()
    torch.testing.assert_close(stable, legacy, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-7)

    # detached weights: grad == w * (softmax - onehot) / n_valid
    shift_logits = logits[:, :-1].reshape(-1, 32)
    shift_labels = labels[:, 1:].reshape(-1)
    mask = shift_labels != -100
    z = shift_logits[mask]
    w = eaft_entropy_weights(torch.topk(z, 8).values, 8, 1.0, True, stable=True)
    g = torch.softmax(z, -1)
    g[torch.arange(len(z)), shift_labels[mask]] -= 1.0
    g = g * w.unsqueeze(1) / mask.sum()
    expected = torch.zeros_like(shift_logits)
    expected[mask] = g
    got = b.grad[:, :-1].reshape(-1, 32)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-7)


def test_eaft_loss_stable_upcasts_bf16_logits():
    torch.manual_seed(4)
    logits = torch.randn(1, 5, 64).bfloat16().requires_grad_(True)
    labels = torch.tensor([[-100, 1, 2, 3, 4]])
    loss = eaft_loss(_Out(logits), labels, alpha=1.0, k=8, stable_ce=True)
    assert loss.dtype == torch.float32
    loss.backward()
    assert logits.grad.dtype == torch.bfloat16
    assert torch.isfinite(logits.grad).all()


def test_entropy_weights_stable_matches_legacy_and_handles_inf():
    torch.manual_seed(5)
    topk = torch.sort(torch.randn(16, 20) * 3, dim=-1, descending=True).values
    legacy = eaft_entropy_weights(topk, 20)
    stable = eaft_entropy_weights(topk, 20, stable=True)
    torch.testing.assert_close(stable, legacy, rtol=1e-5, atol=1e-6)
    padded = torch.tensor([[2.0, 1.0, float("-inf"), float("-inf")]])
    w = eaft_entropy_weights(padded, 4, normalize=False, stable=True)
    p = torch.softmax(torch.tensor([2.0, 1.0]), 0)
    torch.testing.assert_close(w, -(p * p.log()).sum().reshape(1))

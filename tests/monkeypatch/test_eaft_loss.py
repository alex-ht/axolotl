"""CPU tests for the logits-based EAFT reference loss."""

import math

import torch

from axolotl.monkeypatch.loss.eaft import eaft_entropy_weights, eaft_loss


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

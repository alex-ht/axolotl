"""CPU tests for the logits-based EAFT reference loss."""

import torch

from axolotl.monkeypatch.loss.eaft import eaft_loss


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

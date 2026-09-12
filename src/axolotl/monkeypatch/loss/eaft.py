"""
eaft (entropy-aware focal training) loss implementation
weights examples by entropy approximation from top-k logits

Reference: https://github.com/ymxyll/LlamaFactory-EAFT/blob/e2ce19e8efcc226450ee8f2b81dfe4e69f1f945d/src/llamafactory/train/trainer_utils.py
Paper: https://arxiv.org/abs/2601.02151 (K=20, H̃ = H^{top-K} / ln(K), linear gating)
"""

import math

import torch
import torch.nn.functional as F

# Must match ``axolotl.kernels.eaft.MAX_K`` (Triton tl.topk power-of-two cap).
EAFT_MAX_K = 32


def eaft_entropy_weights(
    topk_logits: torch.Tensor,
    k: int,
    alpha: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Detached EAFT token weights from the largest ``k`` logits."""
    k_eff = min(k, topk_logits.shape[-1], EAFT_MAX_K)
    chosen = topk_logits[:, :k_eff]
    probs = F.softmax(chosen, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    if normalize and k_eff > 1:
        entropy = entropy / math.log(k_eff)
    return torch.pow(entropy, alpha)


def eaft_loss(
    outputs, labels, num_items_in_batch=None, alpha=1.0, k=20, normalize=True
):
    """
    compute eaft loss with entropy weighting

    Materializes full logits. CUDA training uses the fused linear kernel
    instead (see ``fused_linear_eaft_loss``); this remains the numerical
    reference and the non-CUDA fallback.

    args:
        outputs: model outputs containing logits
        labels: target labels for computing loss
        num_items_in_batch: for sample packing support
        alpha: exponent for entropy weighting (default 1.0, paper linear gating)
        k: number of top logits for entropy approximation (default 20)
        normalize: divide entropy by ln(k) so weights are in [0, 1] (paper default)
    """
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    shift_logits_view = shift_logits.view(-1, vocab_size)
    shift_labels_view = shift_labels.view(-1)

    mask = shift_labels_view != -100
    if not mask.any():
        return shift_logits_view.sum() * 0.0

    k_eff = min(k, vocab_size, EAFT_MAX_K)

    with torch.no_grad():
        top_k_logits, _ = torch.topk(shift_logits_view[mask].float(), k=k_eff, dim=-1)
        weights = eaft_entropy_weights(top_k_logits, k_eff, alpha, normalize)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(shift_logits_view[mask], shift_labels_view[mask])
    weighted_loss = per_token_loss * weights

    if num_items_in_batch is not None:
        loss = weighted_loss.sum() / num_items_in_batch
    else:
        loss = weighted_loss.mean()

    return loss


def eaft_loss_from_linear(
    hidden,
    weight,
    labels,
    num_items_in_batch=None,
    alpha=1.0,
    k=20,
    bias=None,
    ignore_index=-100,
    shift: bool = True,
    normalize: bool = True,
):
    """EAFT from hidden states + lm_head. CUDA uses the fused kernel."""
    if shift:
        hidden = hidden[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    if hidden.is_cuda:
        from axolotl.kernels.eaft import fused_linear_eaft_loss

        n_items = num_items_in_batch
        if torch.is_tensor(n_items):
            n_items = n_items.item()
        return fused_linear_eaft_loss(
            hidden,
            weight,
            labels,
            alpha=alpha,
            k=k,
            bias=bias,
            ignore_index=ignore_index,
            num_items_in_batch=n_items,
            normalize=normalize,
        )

    logits = F.linear(hidden, weight, bias)
    vocab_size = logits.size(-1)
    logits_view = logits.reshape(-1, vocab_size)
    labels_view = labels.reshape(-1)
    mask = labels_view != ignore_index
    if not mask.any():
        return logits_view.sum() * 0.0

    k_eff = min(k, vocab_size, EAFT_MAX_K)
    with torch.no_grad():
        top_k_logits, _ = torch.topk(logits_view[mask].float(), k=k_eff, dim=-1)
        weights = eaft_entropy_weights(top_k_logits, k_eff, alpha, normalize)

    per_token_loss = F.cross_entropy(
        logits_view[mask], labels_view[mask], reduction="none"
    )
    weighted_loss = per_token_loss * weights
    if num_items_in_batch is not None:
        return weighted_loss.sum() / num_items_in_batch
    return weighted_loss.mean()

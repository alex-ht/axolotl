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
    stable: bool = False,
) -> torch.Tensor:
    """Detached EAFT token weights from the largest ``k`` logits.

    ``stable=True`` computes the entropy from ``log_softmax`` (no epsilon).
    """
    k_eff = min(k, topk_logits.shape[-1], EAFT_MAX_K)
    chosen = topk_logits[:, :k_eff]
    if stable:
        log_probs = F.log_softmax(chosen, dim=-1)
        plogp = torch.exp(log_probs) * log_probs
        # 0 * log 0 := 0 (rows padded with -inf logits)
        plogp = torch.where(torch.isfinite(log_probs), plogp, 0.0)
        entropy = -plogp.sum(dim=-1)
    else:
        probs = F.softmax(chosen, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    if normalize and k_eff > 1:
        entropy = entropy / math.log(k_eff)
    return torch.pow(entropy, alpha)


class StableCE(torch.autograd.Function):
    """Hard-target cross entropy without Softmax Collapse (arXiv:2610.31642).

    With ``m = z_t - LSE_{j!=t}(z_j)`` the per-row loss is ``softplus(-m)``.
    Gradients use the closed form ``g_j = exp(z_j - z_t - L)`` for ``j != t``
    and ``g_t = -sigmoid(-m)`` (= ``-sum_{j!=t} g_j``), so the row gradient
    sums to zero even when ``p_t`` rounds to 1. Computes in fp32 (fp64 if the
    input is fp64) and returns per-row losses; reduction is up to the caller.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor):
        compute_dtype = (
            torch.float64 if logits.dtype == torch.float64 else torch.float32
        )
        z = logits.to(compute_dtype)
        idx = labels.unsqueeze(-1)
        z_t = z.gather(-1, idx).squeeze(-1)
        z_not = z.scatter(-1, idx, float("-inf"))
        lse_not = torch.logsumexp(z_not, dim=-1)
        neg_m = lse_not - z_t
        loss = torch.clamp_min(neg_m, 0.0) + torch.log1p(torch.exp(-torch.abs(neg_m)))
        ctx.save_for_backward(z, labels, z_t, neg_m, loss)
        ctx.in_dtype = logits.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        z, labels, z_t, neg_m, loss = ctx.saved_tensors
        grads = torch.exp(z - (z_t + loss).unsqueeze(-1))
        grads.scatter_(-1, labels.unsqueeze(-1), (-torch.sigmoid(neg_m)).unsqueeze(-1))
        grads = grads * grad_loss.to(grads.dtype).unsqueeze(-1)
        return grads.to(ctx.in_dtype), None


def stable_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-row stable CE (``reduction="none"``) via :class:`StableCE`."""
    return StableCE.apply(logits, labels)


def eaft_loss(
    outputs,
    labels,
    num_items_in_batch=None,
    alpha=1.0,
    k=20,
    normalize=True,
    stable_ce=False,
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
        stable_ce: upcast logits to fp32 and use :class:`StableCE` plus
            log_softmax entropy (``eaft_stable_ce``)
    """
    logits = outputs.logits
    if stable_ce:
        logits = logits.float()

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
        weights = eaft_entropy_weights(
            top_k_logits, k_eff, alpha, normalize, stable=stable_ce
        )

    if stable_ce:
        per_token_loss = stable_cross_entropy(
            shift_logits_view[mask], shift_labels_view[mask]
        )
    else:
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
    stable_ce: bool = False,
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
            stable_ce=stable_ce,
        )

    logits = F.linear(hidden, weight, bias)
    if stable_ce:
        logits = logits.float()
    vocab_size = logits.size(-1)
    logits_view = logits.reshape(-1, vocab_size)
    labels_view = labels.reshape(-1)
    mask = labels_view != ignore_index
    if not mask.any():
        return logits_view.sum() * 0.0

    k_eff = min(k, vocab_size, EAFT_MAX_K)
    with torch.no_grad():
        top_k_logits, _ = torch.topk(logits_view[mask].float(), k=k_eff, dim=-1)
        weights = eaft_entropy_weights(
            top_k_logits, k_eff, alpha, normalize, stable=stable_ce
        )

    if stable_ce:
        per_token_loss = stable_cross_entropy(logits_view[mask], labels_view[mask])
    else:
        per_token_loss = F.cross_entropy(
            logits_view[mask], labels_view[mask], reduction="none"
        )
    weighted_loss = per_token_loss * weights
    if num_items_in_batch is not None:
        return weighted_loss.sum() / num_items_in_batch
    return weighted_loss.mean()

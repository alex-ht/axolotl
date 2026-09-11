"""Fused linear EAFT: lm_head GEMM + online LSE + exact top-k, no [N, V] logits.

Forward keeps a running log-sum-exp and a top-``MAX_K`` buffer while tiling the
vocabulary in SRAM (``tl.dot``). Backward recomputes the same tiles with cuBLAS
and applies detached entropy weights to the CE gradient.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from axolotl.kernels.op_registry import register_kernel_op

# tl.topk requires a power-of-two K. User ``eaft_k`` (default 20) slices this.
MAX_K = 32

_FWD_BLOCK_M = 32
_FWD_BLOCK_N = 128
_FWD_BLOCK_K = 64
_BWD_VOCAB_TILE = 256


def _empty_bias(like: torch.Tensor) -> torch.Tensor:
    return like.new_empty(0)


@triton.jit
def _fused_linear_eaft_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    lse_ptr,
    topk_ptr,
    n_rows,
    n_dim,
    n_vocab,
    stride_xm,
    stride_xd,
    stride_wv,
    stride_wd,
    stride_topk,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOPK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_rows

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    topk = tl.full([BLOCK_M, TOPK_K], -float("inf"), dtype=tl.float32)

    for v_start in range(0, n_vocab, BLOCK_N):
        offs_n = v_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_vocab
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, n_dim, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < n_dim
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xd,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_wv + offs_k[None, :] * stride_wd,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")

        if HAS_BIAS:
            acc += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[
                None, :
            ]

        acc = tl.where(mask_m[:, None] & mask_n[None, :], acc, -float("inf"))

        tile_max = tl.max(acc, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(acc - m_new[:, None]), axis=1)
        m_i = m_new

        tile_top = tl.topk(acc, TOPK_K)
        # interleave, not cat: older Triton has no cat(dim=) and only 1D cat.
        merged = tl.interleave(topk, tile_top)
        topk = tl.topk(merged, TOPK_K)

    lse = m_i + tl.log(d_i)
    # rows with no finite logits (should not happen for valid tokens)
    lse = tl.where(mask_m, lse, 0.0)

    tl.store(lse_ptr + offs_m, lse, mask=mask_m)
    offs_k = tl.arange(0, TOPK_K)
    tl.store(
        topk_ptr + offs_m[:, None] * stride_topk + offs_k[None, :],
        topk,
        mask=mask_m[:, None],
    )


@register_kernel_op("fused_linear_eaft_fwd")
def fused_linear_eaft_fwd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lse, topk)`` with ``topk`` shaped ``[N, MAX_K]``, both fp32."""
    n_rows, n_dim = hidden.shape
    n_vocab = weight.shape[0]
    lse = torch.empty(n_rows, dtype=torch.float32, device=hidden.device)
    topk = torch.empty(n_rows, MAX_K, dtype=torch.float32, device=hidden.device)
    has_bias = bias.numel() > 0
    grid = (triton.cdiv(n_rows, _FWD_BLOCK_M),)
    _fused_linear_eaft_fwd_kernel[grid](
        hidden,
        weight,
        bias if has_bias else hidden,
        lse,
        topk,
        n_rows,
        n_dim,
        n_vocab,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        topk.stride(0),
        HAS_BIAS=has_bias,
        BLOCK_M=_FWD_BLOCK_M,
        BLOCK_N=_FWD_BLOCK_N,
        BLOCK_K=_FWD_BLOCK_K,
        TOPK_K=MAX_K,
        num_warps=4,
        num_stages=2,
    )
    return lse, topk


@fused_linear_eaft_fwd.register_fake
def _(hidden, weight, bias):
    n_rows = hidden.shape[0]
    return (
        hidden.new_empty(n_rows, dtype=torch.float32),
        hidden.new_empty(n_rows, MAX_K, dtype=torch.float32),
    )


@register_kernel_op("fused_linear_eaft_bwd")
def fused_linear_eaft_bwd(
    token_scale: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted CE backward without materializing ``[N, V]``.

    ``token_scale`` is ``dL/dce`` per valid token (already includes entropy
    weights and reduction). Vocabulary is tiled so peak extra memory is
    ``O(N * tile)`` rather than ``O(N * V)``.
    """
    n_vocab = weight.shape[0]
    has_bias = bias.numel() > 0
    hidden_f = hidden.float()
    scale = token_scale.float()
    d_hidden = torch.zeros_like(hidden_f)
    d_weight = torch.zeros_like(weight)
    d_bias = torch.zeros_like(bias) if has_bias else _empty_bias(hidden)

    tile = _BWD_VOCAB_TILE
    for v0 in range(0, n_vocab, tile):
        v1 = min(v0 + tile, n_vocab)
        w_tile = weight[v0:v1].float()
        logits = hidden_f @ w_tile.T
        if has_bias:
            logits = logits + bias[v0:v1].float()
        grads = torch.exp(logits - lse.unsqueeze(1)) * scale.unsqueeze(1)
        in_tile = (labels >= v0) & (labels < v1)
        if in_tile.any():
            grads[in_tile, labels[in_tile] - v0] -= scale[in_tile]
        d_hidden += grads @ w_tile
        d_weight[v0:v1] = (grads.T @ hidden_f).to(dtype=weight.dtype)
        if has_bias:
            d_bias[v0:v1] = grads.sum(0).to(dtype=bias.dtype)

    if not has_bias:
        d_bias = _empty_bias(hidden)
    return d_hidden.to(hidden.dtype), d_weight, d_bias


@fused_linear_eaft_bwd.register_fake
def _(token_scale, hidden, weight, labels, lse, bias):
    d_bias = torch.empty_like(bias) if bias.numel() > 0 else hidden.new_empty(0)
    return torch.empty_like(hidden), torch.empty_like(weight), d_bias


def target_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Indexed GEMM: ``sum_d hidden[n, d] * weight[label[n], d]``."""
    gathered = F.embedding(labels, weight)
    out = (hidden * gathered).sum(dim=-1)
    if bias is not None and bias.numel() > 0:
        out = out + bias[labels].to(dtype=out.dtype)
    return out.float()


def entropy_from_topk(
    topk: torch.Tensor, k: int, alpha: float, normalize: bool = True
) -> torch.Tensor:
    """Detached EAFT weights from the largest ``k`` logits (already fp32)."""
    k_eff = min(k, topk.shape[-1])
    chosen = topk[:, :k_eff]
    probs = torch.softmax(chosen, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    if normalize and k_eff > 1:
        entropy = entropy / math.log(k_eff)
    return torch.pow(entropy, alpha)


def fused_linear_eaft_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    k: int = 20,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    num_items_in_batch: int | float | torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """EAFT loss from hidden states and ``lm_head`` weights (CUDA)."""
    n_items = num_items_in_batch
    if torch.is_tensor(n_items):
        n_items = n_items.item()
    return FusedLinearEAFTFunction.apply(
        hidden,
        weight,
        labels,
        bias if bias is not None else _empty_bias(weight),
        float(alpha),
        int(k),
        bool(normalize),
        int(ignore_index),
        n_items,
    )


class FusedLinearEAFTFunction(torch.autograd.Function):
    """Skip-logits EAFT: Triton LSE/top-k forward, tiled weighted-CE backward."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        bias: torch.Tensor,
        alpha: float,
        k: int,
        normalize: bool,
        ignore_index: int,
        num_items_in_batch,
    ):
        orig_shape = hidden.shape[:-1]
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        labels_flat = labels.reshape(-1).to(device=hidden.device, dtype=torch.long)

        valid = labels_flat != ignore_index
        n_valid = int(valid.sum().item())
        ctx.n_valid = n_valid
        ctx.orig_shape = orig_shape
        ctx.hidden_numel_rows = hidden_flat.shape[0]
        ctx.num_items_in_batch = num_items_in_batch

        if n_valid == 0:
            ctx.save_for_backward(
                hidden_flat,
                weight,
                labels_flat,
                hidden_flat.new_empty(0),
                hidden_flat.new_empty(0),
                bias,
                valid,
            )
            return hidden_flat.sum() * 0.0

        hidden_v = hidden_flat[valid]
        labels_v = labels_flat[valid]
        k_eff = min(int(k), weight.shape[0], MAX_K)

        bias_arg = bias if bias.numel() > 0 else bias
        with torch.no_grad():
            lse, topk = fused_linear_eaft_fwd(
                hidden_v.detach().contiguous(),
                weight.detach().contiguous(),
                bias_arg.detach() if bias_arg.numel() > 0 else bias_arg,
            )
            tgt = target_logits(hidden_v.detach(), weight.detach(), labels_v, bias_arg)
            ce = lse - tgt
            weights = entropy_from_topk(topk, k_eff, float(alpha), bool(normalize))

        ctx.save_for_backward(
            hidden_v,
            weight,
            labels_v,
            lse,
            weights,
            bias,
            valid,
        )

        weighted = ce * weights
        if num_items_in_batch is not None:
            loss = weighted.sum() / float(num_items_in_batch)
        else:
            loss = weighted.mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (
            hidden_v,
            weight,
            labels_v,
            lse,
            weights,
            bias,
            valid,
        ) = ctx.saved_tensors
        if ctx.n_valid == 0:
            return None, None, None, None, None, None, None, None, None

        if ctx.num_items_in_batch is not None:
            denom = ctx.num_items_in_batch
            if torch.is_tensor(denom):
                denom = denom.to(device=grad_output.device, dtype=torch.float32)
            else:
                denom = torch.tensor(
                    denom, device=grad_output.device, dtype=torch.float32
                )
        else:
            denom = torch.tensor(
                ctx.n_valid, device=grad_output.device, dtype=torch.float32
            )

        token_scale = (grad_output.float() * weights.float()) / denom
        d_hidden_v, d_weight, d_bias = fused_linear_eaft_bwd(
            token_scale.contiguous(),
            hidden_v.contiguous(),
            weight.contiguous(),
            labels_v.contiguous(),
            lse.contiguous(),
            bias,
        )

        d_hidden_flat = hidden_v.new_zeros(ctx.hidden_numel_rows, hidden_v.shape[-1])
        d_hidden_flat[valid] = d_hidden_v
        d_hidden = d_hidden_flat.view(*ctx.orig_shape, hidden_v.shape[-1])

        if bias.numel() == 0:
            d_bias = None
        return d_hidden, d_weight, None, d_bias, None, None, None, None, None

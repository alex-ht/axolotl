"""Patch ForCausalLM.forward to compute fused linear EAFT without logits."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.monkeypatch.loss.eaft import eaft_loss_from_linear
from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    from peft.utils import ModulesToSaveWrapper

    _PEFT_AVAILABLE = True
except ImportError:
    ModulesToSaveWrapper = None  # type: ignore[misc, assignment]
    _PEFT_AVAILABLE = False


def _unwrap_lm_head(lm_head):
    if _PEFT_AVAILABLE and isinstance(lm_head, ModulesToSaveWrapper):
        return lm_head.modules_to_save.default
    return lm_head


def _full_tensor(param: torch.Tensor) -> torch.Tensor:
    if hasattr(param, "full_tensor"):
        return param.full_tensor()
    return param


def _eaft_from_lm_head(
    lm_head,
    hidden_states,
    labels,
    shift_labels,
    alpha: float,
    k: int,
    num_items_in_batch=None,
):
    lm_head = _unwrap_lm_head(lm_head)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(lm_head, FullyShardedDataParallel):
            with FullyShardedDataParallel.summon_full_params(lm_head, recurse=False):
                return _eaft_from_lm_head(
                    lm_head.module,
                    hidden_states,
                    labels,
                    shift_labels,
                    alpha,
                    k,
                    num_items_in_batch,
                )
    except ImportError:
        pass

    weight = _full_tensor(lm_head.weight)
    bias = lm_head.bias
    if bias is not None:
        bias = _full_tensor(bias)

    shift = shift_labels is None
    use_labels = labels if shift else shift_labels
    return eaft_loss_from_linear(
        hidden_states,
        weight,
        use_labels,
        num_items_in_batch=num_items_in_batch,
        alpha=alpha,
        k=k,
        bias=bias,
        shift=shift,
    )


def make_eaft_forward(alpha: float, k: int):
    def eaft_forward(
        self,
        *args,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        skip_logits: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        shift_labels = kwargs.pop("shift_labels", None)
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)

        outputs = self.model(
            *args,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        kept_hidden_states = hidden_states[:, slice_indices, :]

        logits = None
        loss = None

        if skip_logits and labels is None and shift_labels is None:
            raise ValueError(
                "skip_logits is True, but labels and shift_labels are None"
            )

        if skip_logits is None:
            skip_logits = self.training and (
                labels is not None or shift_labels is not None
            )

        if skip_logits:
            loss = _eaft_from_lm_head(
                self.lm_head,
                kept_hidden_states,
                labels,
                shift_labels,
                alpha,
                k,
                num_items_in_batch,
            )
        else:
            logits = self.lm_head(kept_hidden_states)
            if labels is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    **kwargs,
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return eaft_forward


def patch_eaft_forward(model_type: str, alpha: float = 1.0, k: int = 20) -> None:
    """Replace ``ForCausalLM.forward`` with a skip-logits EAFT implementation."""
    _, cls_name = get_causal_lm_model_cls_prefix(model_type)
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        module = __import__(module_path, fromlist=[cls_name])
        model_cls = getattr(module, cls_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"Could not patch EAFT forward for model_type={model_type} "
            f"(class {cls_name}): {exc}"
        ) from exc

    model_cls.forward = make_eaft_forward(alpha=alpha, k=k)
    LOG.info(
        "Applied fused linear EAFT forward patch to %s (alpha=%s, k=%s)",
        cls_name,
        alpha,
        k,
    )

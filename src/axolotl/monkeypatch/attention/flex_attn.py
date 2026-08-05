"""Flex attention monkey patch"""

import sys
from typing import Any

import torch
import transformers
from packaging import version

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_KERNEL_OPTIONS_ORIG_ATTR = "_axolotl_flex_kernel_options_original"


def patch_flex_wrapper(**flex_attn_compile_kwargs):
    # TODO remove this patch when transformers#37285 is merged and in a release
    is_torch_2_6 = torch.__version__.startswith("2.6")

    if not is_torch_2_6:
        return

    # Lazy: transformers has since dropped _torch_version from this module's public surface
    # (renamed to get_torch_version), so importing it at module level would break every caller
    # of this file, not just torch-2.6 users. Only this branch (dead on the pinned torch 2.11)
    # still needs it.
    from torch.nn.attention.flex_attention import flex_attention
    from transformers.utils.import_utils import _torch_version, is_torch_less_or_equal

    class WrappedFlexAttention:
        """
        We are doing a singleton class so that flex attention is compiled once when it's first called.
        """

        _instance = None
        _is_flex_compiled = False
        _compiled_flex_attention = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                # Create a new instance if one doesn't already exist
                cls._instance = super().__new__(cls)
            return cls._instance

        @classmethod
        def del_singleton(cls):
            cls._instance = None

        @torch.compiler.disable(recursive=False)
        def __init__(self, training):
            """
            Initialize or update the singleton instance.
            """
            self.training = None
            if not self._is_flex_compiled or training != self.training:
                self.training = training
                if is_torch_less_or_equal("2.5.1"):
                    self._compiled_flex_attention = torch.compile(
                        flex_attention, dynamic=False
                    )
                # In PyTorch 2.6.0, there's a known issue with flex attention compilation which may
                # cause errors. The suggested fix is to compile with "max-autotune-no-cudagraphs"
                # see https://github.com/pytorch/pytorch/issues/146260 for training
                elif version.parse(_torch_version).base_version == "2.6.0" and training:
                    self._compiled_flex_attention = torch.compile(
                        flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                    )
                # Fallback, usually the most recent torch 2.7.x+ versions
                else:
                    LOG.info(
                        "Compiling flex attention with kwargs: %s. This may take a while...",
                        flex_attn_compile_kwargs,
                    )
                    self._compiled_flex_attention = torch.compile(
                        flex_attention,
                        **flex_attn_compile_kwargs,
                    )
                    LOG.info("Flex attention compiled successfully.")

                self._is_flex_compiled = True

        def __call__(self):
            return self._compiled_flex_attention

    transformers.integrations.flex_attention.WrappedFlexAttention = WrappedFlexAttention
    sys.modules[
        "transformers.integrations.flex_attention"
    ].WrappedFlexAttention = WrappedFlexAttention


def patch_flex_kernel_options(kernel_options: dict[str, Any]) -> bool:
    """Force ``kernel_options`` (e.g. ``BLOCK_M``/``BLOCK_N``/``num_stages``/``num_warps``) onto every
    ``flex_attention`` call, overriding Triton's autotuned block sizes. Needed on H100, where the
    autotuned config for some shapes exceeds the 232KB per-SM shared memory limit and flex_attention
    OOMs; smaller blocks (e.g. BLOCK_M=BLOCK_N=16) trade some throughput to fit. Idempotent — a config
    already carrying our marker is left alone rather than double-wrapped."""
    from transformers.integrations.flex_attention import (
        flex_attention_forward as original,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    current = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    if getattr(current, _KERNEL_OPTIONS_ORIG_ATTR, None) is not None:
        return True

    def flex_attention_forward_with_kernel_options(
        module, query, key, value, attention_mask, **kwargs
    ):
        kwargs.setdefault("kernel_options", kernel_options)
        return original(module, query, key, value, attention_mask, **kwargs)

    setattr(
        flex_attention_forward_with_kernel_options, _KERNEL_OPTIONS_ORIG_ATTR, original
    )
    ALL_ATTENTION_FUNCTIONS.register(
        "flex_attention", flex_attention_forward_with_kernel_options
    )
    LOG.info(
        "flex_attn_kernel_options: forcing kernel_options=%s on every flex_attention call",
        kernel_options,
    )
    return True


def unpatch_flex_kernel_options() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    current = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    original = getattr(current, _KERNEL_OPTIONS_ORIG_ATTR, None)
    if original is not None:
        ALL_ATTENTION_FUNCTIONS.register("flex_attention", original)

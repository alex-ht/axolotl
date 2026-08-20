# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA Model-Optimizer examples/gpt-oss/convert_oai_mxfp4_weight_only.py
# https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/gpt-oss/convert_oai_mxfp4_weight_only.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a BF16 GPT-OSS checkpoint back to OpenAI MXFP4 weight-only format."""

from __future__ import annotations

import gc
import json
import os
from typing import Any, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_MODELOPT_INSTALL = (
    "nvidia-modelopt is required to convert GPT-OSS checkpoints to MXFP4. "
    "Install it with: uv pip install --no-build-isolation 'axolotl[modelopt]'"
)


def _require_mxfp4_qtensor():
    try:
        from modelopt.torch.quantization.qtensor import MXFP4QTensor
    except ImportError as exc:
        raise ImportError(_MODELOPT_INSTALL) from exc
    return MXFP4QTensor


def _empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _hf_quant_method(model_name_or_path: str) -> Optional[str]:
    config = AutoConfig.from_pretrained(model_name_or_path)
    qcfg = getattr(config, "quantization_config", None)
    if qcfg is None:
        return None
    if isinstance(qcfg, dict):
        return qcfg.get("quant_method")
    return getattr(qcfg, "quant_method", None)


def to_oai_mxfp4_weight_only(
    model, block_size: int = 32, mxfp4_qtensor=None
) -> dict[str, torch.Tensor]:
    """Quantize GPT-OSS expert weights to OpenAI MXFP4 ``_blocks`` / ``_scales`` tensors."""
    if mxfp4_qtensor is None:
        mxfp4_qtensor = _require_mxfp4_qtensor()

    new_state_dict: dict[str, torch.Tensor] = {}

    for name, param in model.state_dict().items():
        if "experts" in name and "bias" not in name:
            param = param.transpose(-1, -2).contiguous()
            quantized_tensors = []
            scales_tensors = []
            for expert in param:
                quantized, scales = mxfp4_qtensor.quantize(
                    expert, block_size=block_size
                )
                quantized_tensors.append(quantized._quantized_data)
                scales_tensors.append(scales)
            quantized = torch.stack(quantized_tensors)
            scales = torch.stack(scales_tensors)

            shape = quantized.shape
            new_state_dict.update(
                {
                    f"{name}_blocks": quantized.view(
                        shape[0], shape[1], -1, block_size // 2
                    ).cpu(),
                    f"{name}_scales": scales.view(shape[0], shape[1], -1).cpu(),
                }
            )
            del param, quantized, scales
            _empty_device_cache()
        else:
            new_state_dict[name] = param

    return new_state_dict


def write_oai_mxfp4_config(output_path: str) -> None:
    """Stamp HuggingFace MXFP4 ``quantization_config`` onto a converted checkpoint."""
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, encoding="utf-8") as file:
        config_data = json.load(file)

    config_data["quantization_config"] = {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "quant_method": "mxfp4",
    }
    config_data.pop("torch_dtype", None)

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config_data, file, indent=4)


def convert_and_save(model, tokenizer, output_path: str, mxfp4_qtensor=None) -> None:
    quantized_state_dict = to_oai_mxfp4_weight_only(model, mxfp4_qtensor=mxfp4_qtensor)
    model.save_pretrained(output_path, state_dict=quantized_state_dict)
    write_oai_mxfp4_config(output_path)
    tokenizer.save_pretrained(output_path)


def _validate_convert_args(
    model_path: Optional[str],
    lora_path: Optional[str],
    base_path: Optional[str],
) -> None:
    if lora_path and model_path:
        raise ValueError("Specify only one of --model-path or --lora-path, not both.")
    if not lora_path and not model_path:
        raise ValueError("Specify --model-path, or --lora-path with --base-path.")
    if lora_path and not base_path:
        raise ValueError("--base-path is required when --lora-path is set.")


def _load_model(
    model_path: Optional[str],
    lora_path: Optional[str],
    base_path: Optional[str],
    trust_remote_code: bool,
):
    _validate_convert_args(model_path, lora_path, base_path)

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if lora_path:
        from peft import PeftModel

        load_path = base_path
        if _hf_quant_method(base_path) == "mxfp4":
            kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    else:
        load_path = model_path

    LOG.info(f"Loading model from {load_path}.")
    model = AutoModelForCausalLM.from_pretrained(load_path, **kwargs)

    if lora_path:
        LOG.info(f"Merging LoRA adapter from {lora_path}.")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        _empty_device_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        load_path, trust_remote_code=trust_remote_code
    )
    return model, tokenizer


def do_cli(
    output_path: str,
    model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    base_path: Optional[str] = None,
    trust_remote_code: bool = False,
) -> None:
    """Load a GPT-OSS BF16 (or LoRA) checkpoint and export OpenAI MXFP4 weights."""
    _validate_convert_args(model_path, lora_path, base_path)
    mxfp4_qtensor = _require_mxfp4_qtensor()
    model, tokenizer = _load_model(
        model_path=model_path,
        lora_path=lora_path,
        base_path=base_path,
        trust_remote_code=trust_remote_code,
    )
    LOG.info(f"Converting expert weights to MXFP4 and saving to {output_path}.")
    convert_and_save(model, tokenizer, output_path, mxfp4_qtensor=mxfp4_qtensor)
    LOG.info("MXFP4 conversion complete.")

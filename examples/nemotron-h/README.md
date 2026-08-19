# Nemotron-H (nvidia/NVIDIA-Nemotron-3-*)

Hybrid Mamba2 / Attention / MoE architecture (`model_type: nemotron_h`).

| Model | Total params | Active params | Layers |
|---|---|---|---|
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 120B | ~12B | 88 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 30B | ~3B | — |

## Requirements

```bash
pip install mamba-ssm causal-conv1d   # fast Mamba2 CUDA kernels
```

## Architecture notes

- Three block types per layer: **Mamba2** (selective SSM), **Attention** (sparse), **MoE** (mixture-of-experts).
- Only ~12 out of 88 blocks are attention layers (120B variant).
- MLP activation is `relu2` via `mlp_hidden_act` (not the usual `hidden_act`).

## LoRA kernel patches

All three LoRA Triton kernel patches must be disabled:

```yaml
lora_qkv_kernel: false   # attention lives in NemotronHBlock.mixer, not layer.self_attn
lora_o_kernel: false     # same reason
lora_mlp_kernel: false   # relu2 (mlp_hidden_act) is not supported by lora_mlp_kernel
```

## MoE expert weights

NemotronH experts store `up_proj` and `down_proj` as 3D `nn.Parameter` tensors
(shape `[num_experts, out_dim, in_dim]`), **not** `nn.Linear` modules — there is no
`gate_proj`. To fine-tune them alongside attention, use `lora_target_parameters`
instead of `lora_target_modules`:

```yaml
lora_target_parameters:
  - up_proj
  - down_proj
```

## Expert Parallel (DeepEP)

Nemotron-3 Super LatentMoE experts are non-gated 3D `up_proj` / `down_proj` on `moe_latent_size` (1024). DeepEP dispatch/combine uses that latent width, **not** `moe_intermediate_size` (2688). Use grouped_mm; ScatterMoE/SonicMoE are gated-only. `expert_parallel_size` must be ≥ 4 so each rank holds ≤ 128 of the 512 experts.

See `examples/nemotron-h/120b-a12b-ep-fft.yaml`.

## Limitations

- **MoE Triton kernels**: `lora_mlp_kernel` is not supported for NemotronH's MoE expert layers. The expert weights are 3D `nn.Parameter` tensors (not `nn.Linear`), which the Triton kernel does not support. Keep `lora_mlp_kernel: false`.
- **Gradient checkpointing**: Only supported when `sample_packing: true`. Without sample packing the upstream model marks `supports_gradient_checkpointing = False`.

## Training notes

- Do **not** set `trust_remote_code: true`. transformers 5.14+ already registers native `nemotron_h`. The Hub `auto_map` modeling file hard-imports `mamba_ssm` and fails if that package is missing.
- Install local `mamba-ssm` and `causal-conv1d`. Axolotl binds those packages first (and wraps Hub `lazy_load_kernel`) so packing and `Mixer.__init__` do not need `USE_HUB_KERNELS=0`.
- QLoRA automatically leaves `out_proj` and `lm_head` in bf16. The fused Mamba2 kernel and Cut Cross Entropy both read those weights as raw tensors; 4-bit packed storage breaks the shapes.
- transformers 5.14+ names mixers `linear_attention` / `full_attention` / `moe` / `mlp`. The packing patch accepts those names and the older `mamba` / `attention` aliases.
- FSDP2 `offload_params` keeps `NemotronHTopkRouter.e_score_correction_bias` on CPU. Axolotl copies it onto the compute device in the router forward; do not assign the moved tensor back onto the module.
- If the image has `flash-attn-4` but not classic `flash-attn`, set `attn_implementation: flash_attention_4`. Requesting `flash_attention_2` remaps to a Hub kernel name that is unregistered when Hub kernels are disabled.

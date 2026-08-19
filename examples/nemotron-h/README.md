# Nemotron-H (nvidia/NVIDIA-Nemotron-3-* and NVIDIA-Nemotron-3.5-*)

Hybrid Mamba2 / Attention / MoE architecture (`model_type: nemotron_h`).

| Model | Total params | Active params | Layers |
|---|---|---|---|
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 120B | ~12B | 88 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 30B | ~3B | — |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 30B | ~3B | 52 |

## Requirements

```bash
pip install mamba-ssm causal-conv1d   # fast Mamba2 CUDA kernels
```

`sample_packing` and `context_parallel_size > 1` need these kernels: only they
take the `seq_idx` that resets SSM state at packed-sample boundaries, while the
transformers torch fallback drops it silently. Axolotl probes the scan that is
actually in use on the first packed batch and stops rather than mixing state
across samples.

## Architecture notes

- Three block types per layer: **Mamba2** (selective SSM), **Attention** (sparse), **MoE** (mixture-of-experts).
- Only ~12 out of 88 blocks are attention layers (120B variant).
- MLP activation is `relu2` via `mlp_hidden_act` (not the usual `hidden_act`).
- Nemotron-3.5 ships its block pattern as `layers_block_type` (Nemotron-3 used the
  `hybrid_override_pattern` string) and adds multi-token-prediction weights, which
  transformers ignores on load.

## LoRA kernel patches

Attention LoRA kernels support the attention blocks under `NemotronHBlock.mixer`.
Mamba and MoE blocks are skipped by attention patching:

```yaml
lora_qkv_kernel: true
lora_o_kernel: true
lora_mlp_kernel: true    # dense/shared up_proj + down_proj adapters only
```

The MLP option fuses dense/shared-expert LoRA projections and ReLU² activation
forward/backward. Include `up_proj` and `down_proj` in `lora_target_modules` to
adapt these modules. Attention-only examples leave this option disabled.

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

Nemotron-3 Super LatentMoE experts are non-gated 3D `up_proj` / `down_proj` on `moe_latent_size` (1024). DeepEP dispatch/combine uses that latent width, **not** `moe_intermediate_size` (2688). The example uses `grouped_mm`; upstream ScatterMoE also handles the non-gated layout (see `src/axolotl/integrations/kernels/README.md`). `expert_parallel_size` must be ≥ 4 so each rank holds ≤ 128 of the 512 experts.

See `examples/nemotron-h/120b-a12b-ep-fft.yaml`.

## Limitations

- **Routed experts**: `lora_mlp_kernel` only patches dense/shared modules. Routed 3D expert tensors retain their existing PEFT or configured MoE backend; enabling dense MLP kernels does not optimize these routed expert tensors.
- **Sample packing / context parallelism**: requires `mamba-ssm` and `causal-conv1d`; see Requirements above.

## Training notes

- Do **not** set `trust_remote_code: true`. transformers already registers native `nemotron_h`. The Hub `auto_map` modeling file hard-imports `mamba_ssm` and fails if that package is missing.
- Install local `mamba-ssm` and `causal-conv1d` (or set `use_kernels: true`); the transformers torch fallbacks drop `seq_idx` and packing refuses to run without real kernels.
- QLoRA automatically leaves `out_proj` and `lm_head` in bf16. The fused Mamba2 kernel and Cut Cross Entropy both read those weights as raw tensors; 4-bit packed storage breaks the shapes.
- FSDP2 `offload_params` keeps `NemotronHTopkRouter.e_score_correction_bias` on CPU. Axolotl copies it onto the compute device in the router forward; do not assign the moved tensor back onto the module.
- If the image has `flash-attn-4` but not classic `flash-attn`, set `attn_implementation: flash_attention_4`. Requesting `flash_attention_2` remaps to a Hub kernel name that is unregistered when Hub kernels are disabled.

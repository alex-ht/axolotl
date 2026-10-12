# Llama 4 離線訓練映像檔（axolotl v0.13.1 + transformers 4.57.6）

給「完全沒有網路」的叢集使用的 Llama 4 微調映像檔。所有套件、CUDA 擴充與 DeepSpeed ops 都在建置時準備好，執行時不需要連到 PyPI、GitHub 或 Hugging Face Hub。

## 版本選擇

| 元件 | 版本 |
|---|---|
| 基底映像 | `axolotlai/axolotl:0.13.1`（以 digest 固定） |
| Python / CUDA / PyTorch | 3.11 / 12.8 / 2.9.0+cu128 |
| axolotl | 0.13.1（commit `6e42def`，非 editable 安裝） |
| transformers / accelerate / peft / trl | 4.57.6 / 1.12.0 / 0.18.1 / 0.25.1 |
| liger-kernel / torchao / bitsandbytes | 0.6.4 / 0.13.0 / 0.49.1 |
| DeepSpeed | 0.18.2（預先編譯 cpu_adam、fused_adam、async_io） |
| flash-attn / ring-flash-attn | 2.8.3 / 0.1.8 |
| Cut Cross Entropy | axolotl fork `318b7e2` |

為什麼是 v0.13.1：這是 axolotl 最後一個使用 transformers v4（4.57.6）的版本，v0.14.0 之後（包含本 fork 的 HEAD 0.19.0.dev0）都需要 transformers 5.x。本 fork 的 Llama 4 相關程式（`monkeypatch/models/llama4`、`cli/delinearize_llama4.py`、`examples/llama-4`）在 v0.13.1 跟 HEAD 之間完全沒有差異，所以功能上不會少。

## 建置（在有網路的機器上）

```bash
cd ~/Workspace/axolotl
docker build -f docker/Dockerfile-llama4-offline \
    -t axolotl-llama4-offline:0.13.1 docker/llama4-offline
```

可調整的 build arg：

- `TORCH_CUDA_ARCH_LIST`：目標 GPU 架構，預設 `9.0 10.0 12.0+PTX`（H100/H200、B200/GB200、RTX 50 系列）。**已不含 sm_80（A100）**，需要的話請自行加上 `8.0`。只列實際會用到的架構可以縮短編譯時間。
- `BNB_SOURCE`：預設 `pypi`（bitsandbytes 0.49.1，建議維持預設）。要改用自己 fork 的版本，就傳 git branch，例如 `--build-arg BNB_SOURCE=fix/issue-1785`（請先確認這個 branch 能搭配 torch 2.9）。
- `DEEPSPEED_VERSION`、`CCE_REF`、`BASE_IMAGE`。
- `patches/*.patch`：建置時會用 `git apply` 套到 axolotl v0.13.1 原始碼上（預設沒有 patch）。
- `tokenizer/`：如果要把 tokenizer / config 一起包進去（**不含權重**），就把檔案放在這裡，會複製到 `/opt/llama4-tokenizer`。

## 搬到離線主機

```bash
docker save axolotl-llama4-offline:0.13.1 | zstd -T0 -o axolotl-llama4-offline.tar.zst
# 用隨身碟或內部傳輸方式搬過去之後：
zstd -dc axolotl-llama4-offline.tar.zst | docker load
```

## 先跑自我檢查

```bash
docker run --rm --gpus all --network none axolotl-llama4-offline:0.13.1 llama4-offline-check
```

這會檢查離線環境變數、各套件版本、Llama4 類別、axolotl 的 llama4 模組、CCE（含 llama4 patch）、Liger、torchao、flex attention、flash-attn、ring-flash-attn、DeepSpeed 預編 ops、bitsandbytes CUDA、`axolotl --help`，最後用隨機初始化的小型 Llama4 跑一次 forward/backward（CCE + Liger + 編譯過的 flex attention + linearized experts）。

## 執行訓練

權重請自己準備並掛載進來（映像檔裡**不含** Llama 4 權重）：

```
/models   模型權重與 tokenizer（例如 /models/Llama-4-Scout-17B-16E-Linearized-bnb-nf4-bf16）
/data     資料集（範例設定讀 /data/train.jsonl）
/outputs  checkpoint、LoRA adapter、前處理後的資料
```

```bash
docker run --rm -it --gpus all --network none --ipc=host --shm-size 64g \
  -v /path/models:/models:ro -v /path/data:/data:ro -v /path/outputs:/outputs \
  axolotl-llama4-offline:0.13.1 \
  axolotl train /workspace/llama4-configs/scout-qlora-single-gpu-offline.yaml
```

內建的設定檔（`/workspace/llama4-configs/`）：

- `scout-qlora-single-gpu-offline.yaml`：單張 H100、QLoRA、flex attention + torch.compile、CCE、Liger、LoRA kernels、4-bit AdamW、offload gradient checkpointing
- `scout-qlora-fsdp2-offline.yaml`：多卡 FSDP2 + flex attention
- `scout-lora-deepspeed-zero3-offline.yaml`：多卡 bf16 LoRA + DeepSpeed ZeRO-3

多卡請用 `axolotl train ... --launcher accelerate`，或是 `accelerate launch -m axolotl.cli.train ...`。

Linearized 權重要轉回 HF 原生格式：`axolotl delinearize-llama4 --model /outputs/merged --output /outputs/delinearized`。

## 已經處理的離線設定

`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`HF_DATASETS_OFFLINE`、`HF_HUB_DISABLE_TELEMETRY`=1，`WANDB_MODE=offline`，`DO_NOT_TRACK`／`AXOLOTL_DO_NOT_TRACK`=1（關掉 axolotl 的 PostHog 遙測），`RAY_USAGE_STATS_ENABLED=0`，`PIP_NO_INDEX=1` + `PIP_FIND_LINKS=/opt/wheelhouse`。也已經事先寫好 accelerate 的預設設定檔。

`/opt/wheelhouse` 會保留 axolotl、DeepSpeed（預編 ops）、CCE、bitsandbytes 跟大部分相依套件的 wheel，離線時也能用 `pip install --force-reinstall --no-deps <wheel>` 修復環境。torch、nvidia-*、triton、flash-attn 這類大型 wheel 沒有另外收進 wheelhouse，只保留已經安裝在環境裡的版本。

## 已知限制

- 設定檔裡的 `base_model`、`datasets.path` 必須指向本機路徑；如果用 Hub ID 或 `https://` 的設定檔、資料集，離線時一定會失敗。
- 第一次跑 flex attention / torch.compile 時，Triton 會在本機 JIT 編譯（不需要網路，但前幾步會比較慢）。可以把 `/root/.triton` 跟 `/root/.cache/torchinductor` 掛成 volume，重複使用編譯快取。
- 這個 fork 在 HEAD 新增的 MoE kernel（ScatterMoE、SonicMoE、DeepGEMM、DeepEP、expert_parallel）全部需要 transformers 5，**沒有**包進來；這些 kernel 本來也沒有支援 Llama 4。
- 基底映像的 flash-attn 只編譯到 sm_90+PTX。Llama 4 範例本來就建議用 flex attention（見 `examples/llama-4/do-no-use-fa2`）。
- torchao 0.13 的 C++ 擴充跟 torch 2.9 不相容，啟動時會出現 `Skipping import of cpp extensions` 警告。fp8 訓練跟 4/8-bit optimizer 走的是 Python／Triton 路徑，不受影響。
- 驅動程式需要支援 CUDA 12.8（R570 以上）。

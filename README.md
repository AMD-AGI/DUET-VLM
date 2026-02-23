<div align="center">
  <br>
  <br>
  <h1>DUET-VLM: Dual-Stage Efficient Token Reduction for Vision-Language Models</h1>
<a href="https://github.com/AMD-AGI/DUET-VLM"><img src="https://img.shields.io/badge/GitHub-DUET--VLM-blue?logo=github" alt="GitHub"></a>
<a href="https://github.com/AMD-AGI/DUET-VLM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</div>

---

DUET-VLM is a dual-stage token reduction framework that significantly speeds up both training and inference in Vision-Language Models (VLMs). It removes redundant visual tokens while preserving task-critical information, enabling faster iterations and lower serving latency for both image and video reasoning tasks.

Modern VLMs can produce 2,800+ visual tokens from a single high-resolution image, making attention cost the dominant bottleneck. DUET-VLM addresses this by coordinating compression across **both** the vision encoder and the language backbone:

- **Stage 1 — Vision-to-Vision (V2V) Merging**: Uses vision self-attention to identify "dominant" tokens and groups remaining tokens with localized cluster aggregation (VisionZip). This removes background redundancy before it ever hits the LLM.
- **Stage 2 — Text-to-Vision (T2V) Pruning**: Uses salient text tokens to guide layer-wise rank-and-drop pruning of visual tokens inside the LLM (PyramidDrop). This makes token retention context-aware — the model keeps the visual evidence that supports the question being asked.

## Key Results

- **~31% training speedup** with less than 1% accuracy drop
- **>99% baseline accuracy** with 67% fewer tokens at inference
- **Video performance**: matches or exceeds baseline while cutting tokens by ~53%
- At extreme 93.4% token reduction on video, still retains 97.6% accuracy

## Supported Models

| Model | VisionZip Function | PyramidDrop | Config Location |
|-------|-------------------|-------------|-----------------|
| LLaVA-1.5 | `visionzip()` | `modeling_llama_pdrop.py` | `llava/model/` |
| LLaVA-1.6/Next | `visionzip()` | `modeling_llama_pdrop.py` | `llava/model/` |
| Video-LLaVA | `visionzip_video()` | `modeling_llama_pdrop.py` | `videollava/model/` |
| Qwen2.5-VL | Built-in `configure_duet()` | Built-in | `qwen2_5_vl/modeling_qwen2_5vl_duet.py` |

## Getting Started

### Installation

```bash
git clone https://github.com/AMD-AGI/DUET-VLM.git
cd DUET-VLM

# Core LLaVA-1.5 support
pip install -e .

# With Video-LLaVA support (adds decord, einops)
pip install -e ".[video]"

# With Qwen2.5-VL support (adds qwen-vl-utils)
pip install -e ".[qwen]"

# Everything
pip install -e ".[all]"
```

### Dependencies by Model

| Model | Required Packages |
|-------|------------------|
| LLaVA-1.5 | torch, transformers, pillow, accelerate |
| Video-LLaVA | + decord, einops, av |
| Qwen2.5-VL | + qwen-vl-utils |

## Example Usage

### LLaVA-1.5 with DUET-VLM

```python
from llava.model.builder import load_pretrained_model
from visionzip import visionzip

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b"
)

# Apply VisionZip (Stage 1: patches CLIP encoder)
model = visionzip(model, dominant=170, contextual=35, cluster_width=4)

# PyramidDrop (Stage 2) is integrated in the model forward pass
```

### Video-LLaVA with DUET-VLM

```python
from videollava.model.builder import load_pretrained_model
from visionzip import visionzip_video

tokenizer, model, processor, context_len = load_pretrained_model(
    model_path="LanguageBind/Video-LLaVA-7B",
    model_base=None,
    model_name="Video-LLaVA-7B"
)

# Apply VisionZip for Video-LLaVA (patches LanguageBind towers)
model = visionzip_video(model, dominant=170, contextual=35, cluster_width=4)
```

### Qwen2.5-VL with DUET-VLM

```python
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Configure DUET (VisionZip + PyramidDrop)
model.configure_duet(
    visionzip_enabled=True,
    dominant_tokens=170,
    contextual_tokens=35,
    pdrop_enabled=True,
    layer_list=[14, 21],
    ratio_list=[0.5, 0.25]
)
```

## Prerequisites

### Patching Transformers Generation (`utils.py`)

DUET-VLM's Stage 2 (T2V pruning) passes salient text-token indices (`idxs`) through the HuggingFace `model.generate()` pipeline. Stock Transformers does not forward this keyword argument to the model, so a patched `utils.py` must replace the installed copy:

```bash
cp utils.py $(python -c "import transformers; print(transformers.__path__[0])")/generation/
```

> Without this patch, `idxs` is silently dropped and PDrop pruning will not activate during inference.

### Model Consolidation

Training with DeepSpeed produces sharded checkpoint files. Before evaluation, these must be consolidated into a single HuggingFace-compatible checkpoint to avoid key-mismatch errors when `load_pretrained_model()` is called:

```bash
python -m llava.model.consolidate \
    --src /path/to/deepspeed-checkpoint \
    --dst /path/to/consolidated-checkpoint
```

The consolidated path is what you pass as `MODEL_PATH` to the evaluation scripts.

## Evaluation

### Running Benchmarks

Each LLaVA-1.5 eval script accepts three positional arguments:

| Arg | Position | Required? | Default |
|-----|----------|-----------|---------|
| `CUSTOM_NAME` | `$1` | No | `vzpd_cw4_192` (used in output filenames) |
| `MODEL_PATH` | `$2` | No (but should be set for finetuned models) | `liuhaotian/llava-v1.5-7b` |
| `EXTRA_ARGS` | `$3` | No | `--layer_list [16,24] --image_token_ratio_list [0.5,0.0] --dominant 300 --contextual 7 --cluster_width 4 --conv_mode vicuna_v1 --compute_salient_tokens True` |

**Quick run (base model, default hyperparams):**

```bash
bash scripts/llava/v1_5/pdrop_eval/textvqa.sh
```

**Full run with a finetuned model (single benchmark):**

```bash
bash scripts/llava/v1_5/pdrop_eval/textvqa.sh \
    "duet_192" \
    "/path/to/consolidated-checkpoint" \
    "--layer_list [16,24] --image_token_ratio_list [0.5,0.0] --dominant 300 --contextual 7 --cluster_width 4 --conv_mode vicuna_v1 --compute_salient_tokens True"
```

**Full evaluation suite (all benchmarks in parallel across GPUs):**

```bash
VERSION="duet_cw4_192"
CUSTOM_NAME="duet_192"
MODEL_PATH="/path/to/consolidated-checkpoint"
EXTRA_ARGS="--layer_list [16,24] --image_token_ratio_list [0.5,0.0] --dominant 300 --contextual 7 --cluster_width 4 --conv_mode vicuna_v1 --compute_salient_tokens True"

mkdir -p logs/eval
CUDA_VISIBLE_DEVICES=0 bash scripts/llava/v1_5/pdrop_eval/gqa.sh       "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_gqa.log &
CUDA_VISIBLE_DEVICES=1 bash scripts/llava/v1_5/pdrop_eval/sqa.sh       "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_sqa.log &
CUDA_VISIBLE_DEVICES=2 bash scripts/llava/v1_5/pdrop_eval/pope.sh      "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_pope.log &
CUDA_VISIBLE_DEVICES=3 bash scripts/llava/v1_5/pdrop_eval/mme.sh       "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_mme.log &
CUDA_VISIBLE_DEVICES=4 bash scripts/llava/v1_5/pdrop_eval/textvqa.sh   "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_textvqa.log &
CUDA_VISIBLE_DEVICES=5 bash scripts/llava/v1_5/pdrop_eval/mmbench.sh   "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_mmbench.log &
CUDA_VISIBLE_DEVICES=6 bash scripts/llava/v1_5/pdrop_eval/seed.sh      "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_seed.log &
CUDA_VISIBLE_DEVICES=7 bash scripts/llava/v1_5/pdrop_eval/vqav2.sh     "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_vqav2.log &
wait

CUDA_VISIBLE_DEVICES=0 bash scripts/llava/v1_5/pdrop_eval/mmvet.sh     "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_mmvet.log &
CUDA_VISIBLE_DEVICES=1 bash scripts/llava/v1_5/pdrop_eval/llavabench.sh "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_llavabench.log &
CUDA_VISIBLE_DEVICES=2 bash scripts/llava/v1_5/pdrop_eval/vizwiz.sh    "$CUSTOM_NAME" "$MODEL_PATH" "$EXTRA_ARGS" 2>&1 | tee -a logs/eval/${VERSION}_vizwiz.log &
wait
```

`VERSION` is only used for log-file naming and is not passed to the eval scripts themselves.

**Token budget presets (dominant-contextual-cluster_width):**

| Total Tokens | dominant | contextual | cluster_width |
|---|---|---|---|
| 192 | 300 | 7 | 4 |
| 128 | 170 | 32 | 4 |
| 64 | 72 | 30 | 4 |

### LLaVA-1.6/Next (Image)

LLaVA-1.6 evaluation scripts follow the same interface as v1.5 but use `train_mem_pdrop_next.py` and different default layer schedules:

```bash
# Single benchmark
bash scripts/llava/v1_6/pdrop_eval/textvqa.sh

# All 12 benchmarks: vqav2, gqa, sqa, textvqa, vizwiz, pope, mmvet, mme, mmbench, mmbench_cn, llavabench, seed
# Same parallel pattern as LLaVA-1.5 above, replacing v1_5 with v1_6
```

Available benchmarks in `scripts/llava/v1_6/pdrop_eval/`: VQAv2, GQA, SQA, TextVQA, VizWiz, POPE, MM-Vet, MME, MMBench, MMBench-CN, LLaVA-Bench, SEED.

### Video-LLaVA (Image + Video)

Video-LLaVA evaluation covers both image benchmarks and video QA benchmarks.

**Image benchmarks** (evaluating the Video-LLaVA model on image tasks):

```bash
# Available: vqav2, gqa, sqa, textvqa, vizwiz, pope, mmvet, mmbench, llavabench
bash scripts/videollava/v1_5/eval/eval_image_gqa.sh
bash scripts/videollava/v1_5/eval/eval_image_textvqa.sh
bash scripts/videollava/v1_5/eval/eval_image_pope.sh
# ... etc.
```

**Video QA benchmarks** (two-step: inference then GPT-based scoring):

```bash
# Step 1: Generate answers
bash scripts/videollava/v1_5/eval/run_qa_msvd.sh
bash scripts/videollava/v1_5/eval/run_qa_msrvtt.sh
bash scripts/videollava/v1_5/eval/run_qa_tgif.sh
bash scripts/videollava/v1_5/eval/run_qa_activitynet.sh

# Step 2: GPT-based evaluation
bash scripts/videollava/v1_5/eval/eval_qa_msvd.sh
bash scripts/videollava/v1_5/eval/eval_qa_msrvtt.sh
bash scripts/videollava/v1_5/eval/eval_qa_tgif.sh
bash scripts/videollava/v1_5/eval/eval_qa_activitynet.sh
```

**Video-ChatGPT benchmarks** (correctness, detail, contextual, temporal, consistency):

```bash
# Step 1: Generate answers
bash scripts/videollava/v1_5/eval/run_benchmark_1_correctness.sh
bash scripts/videollava/v1_5/eval/run_benchmark_2_detail.sh
bash scripts/videollava/v1_5/eval/run_benchmark_3_contextual.sh
bash scripts/videollava/v1_5/eval/run_benchmark_4_temporal.sh
bash scripts/videollava/v1_5/eval/run_benchmark_5_consistency.sh

# Step 2: GPT-based evaluation
bash scripts/videollava/v1_5/eval/eval_benchmark_1_correctness.sh
bash scripts/videollava/v1_5/eval/eval_benchmark_2_detail.sh
bash scripts/videollava/v1_5/eval/eval_benchmark_3_contextual.sh
bash scripts/videollava/v1_5/eval/eval_benchmark_4_temporal.sh
bash scripts/videollava/v1_5/eval/eval_benchmark_5_consistency.sh
```

> **Note:** Video eval scripts have hardcoded model paths and DUET parameters. Edit `dominant`, `contextual`, `layer_list`, and `image_token_ratio_list` directly in the scripts before running.

### Qwen2.5-VL

```bash
# Available benchmarks: gqa, sqa, textvqa, pope, mme
bash scripts/qwen/gqa.sh duet_640
bash scripts/qwen/sqa.sh duet_640
bash scripts/qwen/textvqa.sh duet_640
bash scripts/qwen/pope.sh duet_640
bash scripts/qwen/mme.sh duet_640
```

Qwen scripts support mode selection via `EXTRA_ARGS`: `--mode duet` (default), `--mode ori_visionzip`, or `--mode baseline`.

### Inference-Only Results (LLaVA-1.5-7B)

| Method | Avg Tokens | Token Reduction | Avg Accuracy (%) |
|---|---|---|---|
| LLaVA-1.5-7B (Baseline) | 576 | 0% | 100.0% |
| VisionZip | 192 | 66.7% | 97.7% |
| PyramidDrop | 192 | 66.7% | 96.4% |
| **DUET-VLM** | **192** | **66.7%** | **99.0%** |
| **DUET-VLM** | **64** | **88.9%** | **95.4%** |

### Video Results (Video-LLaVA-7B)

| Method | Avg Tokens | Token Reduction | Avg Accuracy (%) |
|---|---|---|---|
| Video-LLaVA (Baseline) | 2048 | 0% | 100.0% |
| PyramidDrop | 960 | 53.1% | 100.7% |
| **DUET-VLM** | **960** | **53.1%** | **100.8%** |
| **DUET-VLM** | **136** | **93.4%** | **97.6%** |

## Project Structure

```
DUET-VLM/
├── llava/                      # LLaVA-1.5/1.6 model (image VLM)
├── videollava/                 # Video-LLaVA model (image + video VLM)
├── qwen2_5_vl/                 # Qwen2.5-VL DUET (standalone implementation)
├── visionzip/                  # Shared VisionZip module
├── scripts/
│   ├── llava/
│   │   ├── v1_5/
│   │   │   ├── pdrop_train/    # LLaVA-1.5 training (pretrain, finetune)
│   │   │   └── pdrop_eval/     # LLaVA-1.5 image eval (11 benchmarks)
│   │   └── v1_6/
│   │       ├── pdrop_train/    # LLaVA-1.6 training (pretrain, finetune)
│   │       └── pdrop_eval/     # LLaVA-1.6 image eval (12 benchmarks)
│   ├── videollava/
│   │   └── v1_5/
│   │       ├── pretrain.sh     # Video-LLaVA pre-training
│   │       ├── finetune.sh     # Video-LLaVA fine-tuning (full)
│   │       ├── finetune_lora.sh # Video-LLaVA fine-tuning (LoRA)
│   │       └── eval/           # Video-LLaVA eval (image + video benchmarks)
│   └── qwen/                   # Qwen2.5-VL eval (gqa, sqa, textvqa, pope, mme)
├── setup.py                    # Package installation
├── STRUCTURE.md                # Detailed codebase documentation
└── utils.py                    # Modified HF generation utils
```

## Training

DUET-VLM supports training with integrated token compression. Both pretrain and finetune scripts accept positional arguments for all hyperparameters (with sensible defaults).

### LLaVA-1.5

**Quick run (default hyperparams):**

```bash
# Pre-training (projector only)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava/v1_5/pdrop_train/pretrain.sh

# Fine-tuning (full model)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava/v1_5/pdrop_train/finetune.sh
```

**Full parametrized run:**

```bash
RUN_NAME="duet-192-cw4"
PRETRAIN_OUTPUT_DIR="/path/to/checkpoints/pretrain-${RUN_NAME}"
FINETUNE_OUTPUT_DIR="/path/to/checkpoints/finetune-${RUN_NAME}"
layer_list="[16,24]"
image_token_ratio_list="[0.5,0.0]"
dominant_num="300"
context_num="7"
cluster_width="4"
PRETRAIN_MM_MLP_ADAPTER="${PRETRAIN_OUTPUT_DIR}/mm_projector.bin"

# Stage 1: Pre-training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava/v1_5/pdrop_train/pretrain.sh \
    "$RUN_NAME" "$PRETRAIN_OUTPUT_DIR" "$layer_list" "$image_token_ratio_list" \
    "$dominant_num" "$context_num" "$cluster_width"

# Stage 2: Fine-tuning (pass pretrain adapter as 5th arg)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava/v1_5/pdrop_train/finetune.sh \
    "$RUN_NAME" "$FINETUNE_OUTPUT_DIR" "$layer_list" "$image_token_ratio_list" \
    "$PRETRAIN_MM_MLP_ADAPTER" "$dominant_num" "$context_num" "$cluster_width"

# Consolidate before evaluation
python -m llava.model.consolidate \
    --src "$FINETUNE_OUTPUT_DIR" \
    --dst "${FINETUNE_OUTPUT_DIR}-CONSOLIDATE"
```

Pretrain script args: `$1=RUN_NAME, $2=OUTPUT_DIR, $3=layer_list, $4=image_token_ratio_list, $5=dominant_num, $6=context_num, $7=cluster_width`

Finetune script args: `$1=RUN_NAME, $2=OUTPUT_DIR, $3=layer_list, $4=image_token_ratio_list, $5=PRETRAIN_MM_MLP_ADAPTER, $6=dominant_num, $7=context_num, $8=cluster_width`

### LLaVA-1.6/Next

LLaVA-1.6 uses `train_mem_pdrop_next.py` and the LLaVA-Next architecture with a different default layer schedule (`[8,16,24]`):

```bash
# Pre-training (projector only)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava/v1_6/pdrop_train/pretrain.sh

# Fine-tuning (full model)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/llava/v1_6/pdrop_train/finetune.sh
```

### Video-LLaVA

```bash
# Pre-training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/videollava/v1_5/pretrain.sh

# Fine-tuning (full)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/videollava/v1_5/finetune.sh

# Fine-tuning (LoRA)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/videollava/v1_5/finetune_lora.sh
```

> **Note:** Video-LLaVA training scripts have hardcoded data paths and hyperparameters. Edit them directly to adjust `--dominant_num`, `--context_num`, `--layer_list`, `--image_token_ratio_list`, output directories, and data paths before running.

## Acknowledgement

This codebase builds on [LLaVA](https://github.com/haotian-liu/LLaVA), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [VisionZip](https://github.com/dvlab-research/VisionZip), [PyramidDrop](https://github.com/Cooperx521/PyramidDrop), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).

## License

DUET-VLM is released under the [Apache License 2.0](LICENSE).

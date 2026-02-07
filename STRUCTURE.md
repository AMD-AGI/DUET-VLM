# DUET-VLM Codebase Structure

This document describes the unified DUET-VLM codebase that supports **LLaVA-1.5**, **Video-LLaVA**, and **Qwen2.5-VL** with shared VisionZip + PyramidDrop efficiency techniques.

---

## Quick Reference

| Model | VisionZip Function | PyramidDrop | Config Location |
|-------|-------------------|-------------|-----------------|
| LLaVA-1.5 | `visionzip()` | `modeling_llama_pdrop.py` | `llava/model/` |
| Video-LLaVA | `visionzip_video()` | `modeling_llama_pdrop.py` | `videollava/model/` |
| Qwen2.5-VL | Built-in `configure_duet()` | Built-in | `qwen2_5_vl/modeling_qwen2_5vl_duet.py` |

---

## Directory Structure

```
DUET-VLM/
├── llava/                      # LLaVA-1.5 model (image VLM)
├── videollava/                 # Video-LLaVA model (image + video VLM)
├── qwen2_5_vl/                 # Qwen2.5-VL DUET (standalone implementation)
├── visionzip/                  # Shared VisionZip module
├── scripts/                    # Evaluation and training scripts (see Section 5 for layout)
│   ├── llava/                  # LLaVA-1.5 scripts (USE THESE)
│   ├── videollava/             # Video-LLaVA scripts (USE THESE)
│   ├── qwen/                   # Qwen2.5-VL scripts (USE THESE)
│   ├── v1_5/, v1_6/            # Legacy duplicates of scripts/llava/ — ignore these
│   ├── convert_*.py            # Dataset format conversion utilities
│   ├── zero2.json, zero3.json  # DeepSpeed configs for training
│   └── *.sh                    # Legacy loose training scripts — ignore these
├── setup.py                    # Package installation
└── utils.py                    # Modified HF generation utils (4.6k lines, not a small helper)
```

---

## 1. `llava/` - LLaVA-1.5 Model

The original LLaVA-1.5 implementation with VisionZip + PyramidDrop support.

```
llava/
├── __init__.py
├── constants.py                 # IMAGE_TOKEN_INDEX, IGNORE_INDEX, etc.
├── conversation.py              # Conversation templates (vicuna, llama2, etc.)
├── mm_utils.py                  # Image processing, tokenization helpers
├── salient_token_finder.py      # NLP pipeline to extract salient words (spacy/nltk)
├── aligner.py                   # Greedy alignment between salient words and visual tokens
├── model/
│   ├── __init__.py
│   ├── builder.py               # load_pretrained_model()
│   ├── llava_arch.py            # LlavaMetaModel, LlavaMetaForCausalLM
│   ├── modeling_llama_pdrop.py  # PyramidDrop LLaMA implementation
│   ├── language_model/
│   │   ├── llava_llama.py       # LlavaLlamaForCausalLM
│   │   └── llava_llama_pdrop.py # LlavaConfig with use_salient_tokens
│   └── multimodal_encoder/
│       ├── builder.py           # build_vision_tower()
│       └── clip_encoder.py      # CLIPVisionTower
├── eval/                        # Evaluation scripts
│   ├── model_vqa_loader.py      # Main VQA inference (TextVQA, GQA, etc.)
│   ├── model_vqa_science.py     # ScienceQA evaluation
│   ├── model_vqa_mmbench.py     # MMBench evaluation
│   ├── model_vqa.py             # Basic VQA inference
│   ├── eval_textvqa.py          # TextVQA accuracy calculation
│   ├── eval_pope.py             # POPE accuracy calculation
│   ├── eval_science_qa.py       # ScienceQA accuracy calculation
│   ├── run_llava.py             # Interactive CLI inference
│   └── ...
├── train/
│   ├── pdrop_train.py           # Training with VisionZip + PyramidDrop
│   └── ...
└── serve/                       # Gradio demo, CLI
```

### Key Files

| File | Purpose |
|------|---------|
| `model/builder.py` | `load_pretrained_model()` - loads LLaVA model |
| `model/llava_arch.py` | `prepare_inputs_labels_for_multimodal()` - processes image tokens |
| `model/modeling_llama_pdrop.py` | `pdrop_forward()`, `pdrop_rank_drop()` - PyramidDrop logic |
| `model/language_model/llava_llama_pdrop.py` | `LlavaConfig` with `use_salient_tokens` |
| `salient_token_finder.py` | Extracts question-relevant words via spacy/nltk for salient token selection |
| `aligner.py` | Greedy sequential matcher: aligns extracted words to visual token text |
| `eval/model_vqa_loader.py` | Main evaluation script for benchmarks |
| `eval/model_vqa_loader_MOD.py` | **Modified** VQA loader (experimental variant — see note below) |

> **`model_vqa_loader.py` vs `model_vqa_loader_MOD.py`**: Use `model_vqa_loader.py` for standard evaluations. The `_MOD` variant is an experimental fork with modifications; check its diff if you need to understand what changed.

### Evaluation Scripts

The `eval/` directory contains two types of scripts:

**Inference scripts** (generate predictions):
| Script | Benchmark | Output |
|--------|-----------|--------|
| `model_vqa_loader.py` | TextVQA, GQA, VizWiz, SEED | `.jsonl` with predictions |
| `model_vqa_science.py` | ScienceQA | `.jsonl` with predictions |
| `model_vqa_mmbench.py` | MMBench | `.jsonl` with predictions |
| `model_vqa.py` | Generic VQA | `.jsonl` with predictions |
| `run_llava.py` | Interactive CLI | Terminal output |

**Accuracy calculation scripts** (evaluate predictions):
| Script | Input | Output |
|--------|-------|--------|
| `eval_textvqa.py` | predictions + annotations | Accuracy % |
| `eval_pope.py` | predictions + annotations | Accuracy, F1, Yes ratio |
| `eval_science_qa.py` | predictions + annotations | Accuracy % |

### Usage

```python
from llava.model.builder import load_pretrained_model
from visionzip import visionzip

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b"
)

# Apply VisionZip (patches CLIP encoder)
model = visionzip(model, dominant=191, contextual=30, cluster_width=4)
```

---

## 2. `videollava/` - Video-LLaVA Model

Video-LLaVA extends LLaVA with video understanding via LanguageBind encoders.

```
videollava/
├── __init__.py
├── constants.py                 # Same as llava + video tokens
├── conversation.py
├── mm_utils.py                  # Image + video processing
├── model/
│   ├── __init__.py
│   ├── builder.py               # load_pretrained_model() for Video-LLaVA
│   ├── llava_arch.py            # Handles both image and video towers
│   ├── modeling_llama_pdrop.py  # PyramidDrop for Video-LLaVA
│   ├── language_model/
│   │   └── llava_llama.py       # LlavaLlamaForCausalLM (video-aware)
│   └── multimodal_encoder/
│       ├── builder.py
│       ├── clip_encoder.py
│       └── languagebind/        # LanguageBind encoders
│           ├── __init__.py
│           ├── image/           # LanguageBindImageTower
│           │   ├── modeling_image.py
│           │   └── ...
│           └── video/           # LanguageBindVideoTower
│               ├── modeling_video.py
│               └── ...
├── eval/
│   ├── model_vqa_loader.py             # Image VQA inference (same as llava)
│   ├── eval_textvqa.py                 # Image TextVQA accuracy
│   ├── eval_pope.py                    # Image POPE accuracy
│   ├── eval_gqa.py                     # Image GQA accuracy
│   ├── eval_science_qa.py              # Image ScienceQA accuracy
│   ├── video/                          # Video-specific evaluation
│   │   ├── run_inference_video_qa.py       # Video QA inference (MSVD, MSRVTT, ActivityNet, TGIF)
│   │   ├── eval_video_qa.py               # GPT-based video QA scoring
│   │   ├── run_inference_benchmark_general.py  # Video benchmark inference
│   │   ├── run_inference_benchmark_consistency.py
│   │   ├── eval_benchmark_1_correctness.py     # GPT-based benchmark scoring
│   │   ├── eval_benchmark_2_detailed_orientation.py
│   │   ├── eval_benchmark_3_context.py
│   │   ├── eval_benchmark_4_temporal.py
│   │   └── eval_benchmark_5_consistency.py
│   └── ...
└── train/
    ├── train.py
    └── pdrop_train.py           # VisionZip + PyramidDrop training
```

### Key Differences from LLaVA-1.5

1. **Dual Towers**: Has both `image_tower` (LanguageBindImage) and `video_tower` (LanguageBindVideo)
2. **LanguageBind**: Uses LanguageBind CLIP variants instead of standard OpenAI CLIP
3. **Video Input**: Processes video frames with temporal encoding

> **Note on LanguageBind modalities**: The `languagebind/` directory also contains `audio/`, `depth/`, and `thermal/` modality encoders from the upstream LanguageBind repo. These are **not used** by Video-LLaVA or DUET — only `image/` and `video/` are active.

### Usage

```python
from videollava.model.builder import load_pretrained_model
from visionzip import visionzip_video  # Note: visionzip_video, not visionzip

tokenizer, model, processor, context_len = load_pretrained_model(
    model_path="LanguageBind/Video-LLaVA-7B",
    model_base=None,
    model_name="Video-LLaVA-7B"
)

# Apply VisionZip for Video-LLaVA (patches LanguageBind towers)
model = visionzip_video(model, dominant=191, contextual=30, cluster_width=4)
```

---

## 3. `qwen2_5_vl/` - Qwen2.5-VL DUET

Standalone Qwen2.5-VL implementation with built-in VisionZip + PyramidDrop.

```
qwen2_5_vl/
├── __init__.py                          # Exports model classes
├── modeling_qwen2_5vl_duet.py           # Main model with DUET support
├── eval_benchmarks.py                   # Benchmark evaluation (POPE, GQA, SQA, MME, TextVQA)
├── run_inference.py                     # Single image inference / quick testing
└── README.md                            # Qwen2.5-VL specific documentation
```

### Key Features

- **Built-in DUET**: VisionZip + PyramidDrop implemented directly in model class
- **configure_duet() method**: Easy configuration without monkey-patching
- **3D Position Embeddings**: Handles Qwen2.5-VL's (temporal, height, width) positions

### Usage

```python
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor

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
    layer_list=[7, 14, 21],        # Qwen2.5-VL has 28 layers
    ratio_list=[1.0, 0.75, 0.5, 0.25]
)
```

---

## 4. `visionzip/` - Shared VisionZip Module

The core VisionZip implementation shared across LLaVA-1.5 and Video-LLaVA.

```
visionzip/
├── __init__.py              # Exports visionzip, visionzip_video
├── main.py                  # Main entry points
├── utils.py                 # CLIPAttention_forward, CLIP_EncoderLayer_forward
├── clip_encoder_cw.py       # CLIPVisionTower_VisionZip.forward
├── llava_arch.py            # LLaVA-1.5 specific functions
└── videollava_arch.py       # Video-LLaVA specific functions
```

### Main Functions

| Function | Model | What it patches |
|----------|-------|-----------------|
| `visionzip(model, ...)` | LLaVA-1.5 | CLIP encoder, LlavaMetaForCausalLM |
| `visionzip_video(model, ...)` | Video-LLaVA | LanguageBind image/video towers |

### Parameters

```python
visionzip(
    model,
    dominant=191,      # Number of dominant (high-attention) tokens
    contextual=30,     # Number of contextual (clustered) tokens
    cluster_width=4    # Clustering window size
)
```

### How VisionZip Works

1. **Dominant Token Selection**: Select top-k tokens by CLS attention
2. **Contextual Token Clustering**: Cluster remaining tokens spatially
3. **Token Reduction**: 576 → ~221 tokens (for dominant=191, contextual=30)

---

## 5. `scripts/` - Evaluation & Training Scripts

Organized by model type. **Use the model-specific subdirectories** (`llava/`, `videollava/`, `qwen/`).

> **Warning**: The `scripts/` directory also contains legacy duplicates and loose files from before the codebase was unified. See the "Legacy / Ignore" section below.

```
scripts/
├── llava/                   # ✅ LLaVA-1.5 scripts (USE THESE)
│   ├── v1_5/
│   │   ├── pdrop_eval/      # Evaluation with PyramidDrop
│   │   │   ├── textvqa.sh
│   │   │   ├── sqa.sh
│   │   │   ├── pope.sh
│   │   │   ├── gqa.sh
│   │   │   ├── mme.sh
│   │   │   ├── mmvet.sh
│   │   │   ├── mmbench.sh
│   │   │   ├── mmbench_cn.sh
│   │   │   ├── seed.sh
│   │   │   ├── vizwiz.sh
│   │   │   ├── vqav2.sh
│   │   │   └── llavabench.sh
│   │   └── pdrop_train/
│   │       ├── pretrain.sh
│   │       └── finetune.sh
│   └── v1_6/
│       ├── pdrop_eval/      # Same benchmarks as v1_5
│       └── pdrop_train/
│
├── videollava/              # ✅ Video-LLaVA scripts (USE THESE)
│   ├── v1_5/
│   │   ├── eval/
│   │   │   ├── eval_qa_msvd.sh
│   │   │   ├── eval_qa_msrvtt.sh
│   │   │   ├── eval_qa_activitynet.sh
│   │   │   ├── eval_qa_tgif.sh
│   │   │   ├── eval_image_textvqa.sh
│   │   │   ├── eval_image_sqa.sh
│   │   │   ├── eval_image_pope.sh
│   │   │   ├── eval_image_gqa.sh
│   │   │   ├── eval_image_vqav2.sh
│   │   │   ├── eval_image_vizwiz.sh
│   │   │   ├── eval_image_mmbench.sh
│   │   │   ├── eval_image_mmvet.sh
│   │   │   ├── eval_image_llavabench.sh
│   │   │   ├── run_qa_msvd.sh          # Inference-only (no eval)
│   │   │   ├── run_qa_msrvtt.sh
│   │   │   ├── run_qa_activitynet.sh
│   │   │   ├── run_qa_tgif.sh
│   │   │   └── run_benchmark_*.sh      # Video benchmark inference
│   │   ├── finetune.sh
│   │   └── pretrain.sh
│   ├── finetune.sh          # Top-level training scripts
│   ├── pretrain.sh
│   ├── zero2.json           # DeepSpeed configs for Video-LLaVA training
│   ├── zero3.json
│   └── convert_*.py         # Dataset conversion utils
│
├── qwen/                    # ✅ Qwen2.5-VL scripts (USE THESE)
│   ├── textvqa.sh           # TextVQA evaluation
│   ├── pope.sh              # POPE evaluation
│   ├── sqa.sh               # ScienceQA evaluation
│   ├── gqa.sh               # GQA evaluation
│   └── mme.sh               # MME evaluation
│   ├── sweep_duet_token_budgets.sh
│   ├── sweep_visionzip_2880.sh
│   ├── compare_duet_vs_visionzip_orig_2880.sh
│   ├── run_latency_comparison_2880.sh
│   └── reevaluate_results.py
│
├── zero2.json               # DeepSpeed stage 2 config
├── zero3.json               # DeepSpeed stage 3 config
├── zero3_offload.json       # DeepSpeed stage 3 + CPU offload
├── convert_*.py             # Dataset format converters (GQA, MMBench, SQA, etc.)
├── extract_mm_projector.py  # Extract projector weights from checkpoint
├── merge_lora_weights.py    # Merge LoRA adapter into base model
│
│  # ⚠️ LEGACY — duplicates of scripts/llava/, ignore these:
├── v1_5/                    # Duplicate of scripts/llava/v1_5/
├── v1_6/                    # Duplicate of scripts/llava/v1_6/
├── finetune*.sh             # Loose training scripts (pre-unification)
├── pretrain*.sh
└── sqa_eval_*.sh
```

### DeepSpeed Configs

| Config | Use Case |
|--------|----------|
| `zero2.json` | Stage 2 — fastest, needs more GPU memory |
| `zero3.json` | Stage 3 — slower, less GPU memory |
| `zero3_offload.json` | Stage 3 + CPU offload — for limited GPU memory |
| `videollava/zero*.json` | Same configs duplicated for Video-LLaVA training scripts |

### Running Evaluations

```bash
# LLaVA-1.5 TextVQA
cd DUET-VLM
bash scripts/llava/v1_5/pdrop_eval/textvqa.sh

# Video-LLaVA MSVD
bash scripts/videollava/v1_5/eval/eval_qa_msvd.sh

# Qwen2.5-VL TextVQA
bash scripts/qwen/textvqa.sh duet_640

# Qwen2.5-VL POPE
bash scripts/qwen/pope.sh duet_640
```

---

## 6. Installation

```bash
cd DUET-VLM

# Core LLaVA-1.5 support only
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

---

## 7. Common Workflows

### Evaluating LLaVA-1.5 with VisionZip + PyramidDrop

```bash
# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run TextVQA evaluation
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /path/to/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /path/to/textvqa/train_images \
    --answers-file ./answers/textvqa_duet.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# Calculate accuracy
python -m llava.eval.eval_textvqa \
    --annotation-file /path/to/TextVQA_0.5.1_val.json \
    --result-file ./answers/textvqa_duet.jsonl
```

### Evaluating Video-LLaVA

```bash
python videollava/eval/video/run_inference_video_qa.py \
    --model-path LanguageBind/Video-LLaVA-7B \
    --video-folder /path/to/videos \
    --question-file /path/to/questions.json \
    --answers-file ./answers/videollava_duet.jsonl
```

### Evaluating Qwen2.5-VL DUET

```bash
# Per-benchmark (same pattern as LLaVA-1.5):
#   bash scripts/qwen/<benchmark>.sh <CUSTOM_NAME> [MODEL_PATH] [EXTRA_ARGS]

# DUET with default config
bash scripts/qwen/textvqa.sh duet_640

# VisionZip-only (no PyramidDrop)
bash scripts/qwen/textvqa.sh vz_orig_640 Qwen/Qwen2.5-VL-7B-Instruct "--mode ori_visionzip --dominant 540 --contextual 100"

# Baseline (no token reduction)
bash scripts/qwen/textvqa.sh baseline Qwen/Qwen2.5-VL-7B-Instruct "--mode baseline"
```

---

## 8. Key Configuration Options

### Salient Tokens (LLaVA-1.5)

```python
# In model config
model.config.use_salient_tokens = True  # or False

# In training
python llava/train/pdrop_train.py --use_salient_tokens True
```

### PyramidDrop Layer/Ratio Lists

| Model | Default Layers | Ratios |
|-------|---------------|--------|
| LLaVA-1.5 (32 layers) | `[8, 16, 24]` | `[1.0, 0.75, 0.5, 0.25]` |
| Qwen2.5-VL (28 layers) | `[7, 14, 21]` | `[1.0, 0.75, 0.5, 0.25]` |

Ratio list meaning:
- Before layer 8: Keep 100% of visual tokens
- After layer 8: Keep 75%
- After layer 16: Keep 50%
- After layer 24: Keep 25%

---

## 9. File Cross-Reference

| Need to... | Go to... |
|------------|----------|
| Load LLaVA-1.5 model | `llava/model/builder.py` |
| Load Video-LLaVA model | `videollava/model/builder.py` |
| Apply VisionZip to LLaVA | `visionzip/main.py::visionzip()` |
| Apply VisionZip to Video-LLaVA | `visionzip/main.py::visionzip_video()` |
| Configure Qwen2.5-VL DUET | `qwen2_5_vl/modeling_qwen2_5vl_duet.py::configure_duet()` |
| Understand PyramidDrop logic | `llava/model/modeling_llama_pdrop.py::pdrop_rank_drop()` |
| Run LLaVA eval | `llava/eval/model_vqa_loader.py` |
| Run Video-LLaVA eval | `videollava/eval/video/run_inference_video_qa.py` |
| Run Qwen2.5-VL eval | `qwen2_5_vl/eval_benchmarks.py` |

---

## 10. Troubleshooting

### ImportError: No module named 'transformers'
```bash
pip install transformers>=4.37
```

### VisionZip not reducing tokens
Make sure you call `visionzip()` or `visionzip_video()` AFTER loading the model and BEFORE inference.

### Qwen2.5-VL position embedding errors
Ensure you're using `modeling_qwen2_5vl_duet.py` which handles 3D position embeddings correctly with PyramidDrop.

### Video-LLaVA frame count mismatch
LanguageBind expects 8 frames by default. Adjust `num_frames` in config if needed.

---

## Contact

For questions about this codebase, refer to:
- `project_status.md` - Current state and recent changes
- `DEVELOPMENT_LOG.md` - Detailed change history and experiments

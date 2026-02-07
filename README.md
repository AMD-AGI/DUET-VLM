# DUET-VLM

<p align="center">
  <b>DUET-VLM: Dual-Stage Efficient Token Reduction for Vision-Language Models</b>
</p>

<p align="center">
  <a href="https://github.com/AMD-AGI/DUET-VLM"><img src="https://img.shields.io/badge/GitHub-DUET--VLM-blue?logo=github" alt="GitHub"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>

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
model = visionzip(model, dominant=191, contextual=30, cluster_width=4)

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
model = visionzip_video(model, dominant=191, contextual=30, cluster_width=4)
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
    layer_list=[7, 14, 21],
    ratio_list=[1.0, 0.75, 0.5, 0.25]
)
```

## Evaluation

### Running Benchmarks

```bash
# LLaVA-1.5 TextVQA
bash scripts/llava/v1_5/pdrop_eval/textvqa.sh

# Video-LLaVA MSVD
bash scripts/videollava/v1_5/eval/eval_qa_msvd.sh

# Qwen2.5-VL TextVQA
bash scripts/qwen/textvqa.sh duet_640

# Qwen2.5-VL POPE
bash scripts/qwen/pope.sh duet_640
```

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
├── llava/                      # LLaVA-1.5 model (image VLM)
├── videollava/                 # Video-LLaVA model (image + video VLM)
├── qwen2_5_vl/                 # Qwen2.5-VL DUET (standalone implementation)
├── visionzip/                  # Shared VisionZip module
├── scripts/                    # Evaluation and training scripts
│   ├── llava/                  # LLaVA-1.5 scripts
│   ├── videollava/             # Video-LLaVA scripts
│   └── qwen/                   # Qwen2.5-VL scripts
├── setup.py                    # Package installation
├── STRUCTURE.md                # Detailed codebase documentation
└── utils.py                    # Modified HF generation utils
```

## Training

DUET-VLM supports training with integrated token compression. See the training scripts for each model:

```bash
# LLaVA-1.5 pre-training
bash scripts/llava/v1_5/pdrop_train/pretrain.sh

# LLaVA-1.5 fine-tuning
bash scripts/llava/v1_5/pdrop_train/finetune.sh

# Video-LLaVA fine-tuning
bash scripts/videollava/v1_5/finetune.sh
```

## License

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Acknowledgement

This codebase builds on [LLaVA](https://github.com/haotian-liu/LLaVA), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [VisionZip](https://github.com/dvlab-research/VisionZip), [PyramidDrop](https://github.com/Cooperx521/PyramidDrop), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).

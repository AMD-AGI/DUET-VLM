# DUET-VLM for Qwen2.5-VL

**DUET-VLM: Dual-stage Efficient Token reduction for Vision-Language Models**

This implementation combines two complementary token reduction techniques:
1. **VisionZip**: Token clustering at the vision encoder level
2. **PyramidDrop**: Progressive token dropping at the LLM decoder level

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DUET-VLM Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: VisionZip (Vision Encoder)                                        │
│  ═══════════════════════════════════                                        │
│  Input: N visual tokens from ViT                                            │
│  ↓                                                                          │
│  1. Extract attention from last vision block                                │
│  2. Select dominant tokens (top-K by CLS attention)                         │
│  3. Cluster remaining tokens → contextual tokens                            │
│  ↓                                                                          │
│  Output: ~70% reduction (e.g., 1000 → 300 tokens)                           │
│                                                                             │
│  Stage 2: PyramidDrop (LLM Decoder)                                         │
│  ════════════════════════════════════                                       │
│  At specified layers (e.g., 7, 14, 21):                                     │
│  1. Compute text-to-image attention                                         │
│  2. Rank image tokens by attention importance                               │
│  3. Drop lowest-attention tokens                                            │
│  ↓                                                                          │
│  Progressive: 300 → 225 → 150 → 75 tokens                                   │
│                                                                             │
│  Total reduction: ~92.5% (1000 → 75 tokens)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

The implementation requires the same dependencies as Qwen2.5-VL:

```bash
pip install transformers>=4.49.0 torch pillow
pip install flash-attn --no-build-isolation  # Recommended for performance
```

## Quick Start

### Single Image Inference

```bash
# Base Qwen2.5-VL (no token reduction)
python run_inference.py \
    --image path/to/image.jpg \
    --query "What is in this image?"

# VisionZip only (~70% reduction at vision encoder)
python run_inference.py \
    --image path/to/image.jpg \
    --query "What is in this image?" \
    --visionzip \
    --dominant 0.65 \
    --contextual 0.05

# DUET-VLM (VisionZip + PyramidDrop)
python run_inference.py \
    --image path/to/image.jpg \
    --query "What is in this image?" \
    --visionzip \
    --dominant 0.65 \
    --contextual 0.05 \
    --pdrop \
    --layer-list "[7,14,21]" \
    --ratio-list "[0.75,0.5,0.25]"
```

### Benchmark Evaluation

```bash
# Evaluate on TextVQA
python eval_benchmark.py \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --question-file /path/to/textvqa_val.jsonl \
    --image-folder /path/to/textvqa/images \
    --answers-file results/textvqa_duet.jsonl \
    --visionzip \
    --pdrop \
    --layer-list "[7,14,21]" \
    --ratio-list "[0.75,0.5,0.25]"
```

### Compare Models

```bash
# Run full comparison (Base vs VisionZip vs DUET-VLM)
./compare_models.sh \
    --question-file /path/to/textvqa_val.jsonl \
    --image-folder /path/to/textvqa/images \
    --output-dir ./results
```

## Configuration

### VisionZip Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dominant` | 0.65 | Fraction of high-attention tokens to keep |
| `--contextual` | 0.05 | Fraction of clustered context tokens |
| `--cluster-width` | 4 | Preselection multiplier for clustering |

### PyramidDrop Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--layer-list` | None | LLM layers to drop at (e.g., `[7,14,21]`) |
| `--ratio-list` | None | Keep ratios after each layer (e.g., `[0.75,0.5,0.25]`) |

**Note**: Qwen2.5-VL-7B has 28 decoder layers. Recommended drop layers: `[7, 14, 21]` (at 25%, 50%, 75% of model depth).

## Python API

```python
from modeling_qwen2_5vl_duet import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Configure DUET-VLM
model.configure_duet(
    visionzip_enabled=True,
    dominant_ratio=0.65,
    contextual_ratio=0.05,
    pdrop_enabled=True,
    layer_list=[7, 14, 21],
    image_token_ratio_list=[0.75, 0.5, 0.25],
)

# Run inference
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
```

## Expected Performance

### Token Reduction

| Stage | Tokens | Reduction |
|-------|--------|-----------|
| Original (ViT output) | 1000 | - |
| After VisionZip | 700 | 30% |
| After Layer 7 | 525 | 47.5% |
| After Layer 14 | 350 | 65% |
| After Layer 21 | 175 | 82.5% |

### Speedup

- **Prefill**: ~2-3x faster (fewer tokens to process)
- **Memory**: ~40-60% reduction in KV cache
- **Quality**: Minimal accuracy drop (<1-2% on most benchmarks)

## File Structure

```
Qwen2_5_VL_DUET/
├── modeling_qwen2_5vl_duet.py  # Main model implementation
├── run_inference.py            # Single image inference
├── eval_benchmark.py           # Benchmark evaluation
├── compare_models.sh           # Comparison script
└── README.md                   # This file
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{visionzip2024,
  title={VisionZip: Longer is Better but Not Necessary in Vision Language Models},
  author={Yang, Senqiao and others},
  year={2024}
}

@article{pyramiddrop2024,
  title={PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction},
  author={...},
  year={2024}
}
```

## License

Apache 2.0 License

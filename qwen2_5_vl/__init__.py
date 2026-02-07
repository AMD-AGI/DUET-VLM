"""
Qwen2.5-VL DUET Module

DUET-VLM (VisionZip + PyramidDrop) implementation for Qwen2.5-VL models.
This module provides efficient visual token reduction for Qwen2.5-VL.

Key Features:
- VisionZip (Stage 1): Vision encoder token reduction via attention-based clustering
- PyramidDrop (Stage 2): Progressive token dropping at LLM decoder layers
- Dynamic resolution support via min_pixels/max_pixels
- Self-contained model with configure_duet() method

Usage:
    from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Required for VisionZip
    )
    
    # Configure DUET
    model.configure_duet(
        visionzip_enabled=True,
        dominant_ratio=0.65,
        contextual_ratio=0.05,
        pdrop_enabled=True,
        layer_list=[14, 21],
        image_token_ratio_list=[0.5, 0.0],
    )
"""

from .modeling_qwen2_5vl_duet import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
)

# Re-export config from transformers for convenience
try:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLVisionConfig,
    )
except ImportError:
    pass

__all__ = [
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
    "Qwen2_5_VisionTransformerPretrainedModel",
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
]

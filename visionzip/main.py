from .utils import CLIP_EncoderLayer_forward, CLIPAttention_forward, apply_info
import types
from .clip_encoder_cw import CLIPVisionTower_VisionZip
from .llava_arch import prepare_inputs_labels_for_multimodal_visionzip, encode_images_visionzip, encode_images_visionzip_multi, restore_image_features_sorted


def visionzip(model, dominant=191, contextual=30, cluster_width=4):
    """
    Apply VisionZip to LLaVA-1.5 models.
    
    Args:
        model: LLaVA model instance
        dominant: Number of dominant (high-attention) tokens to keep
        contextual: Number of contextual (clustered) tokens to keep
        cluster_width: Multiplier for preselection pool size
    
    Returns:
        model: Modified model with VisionZip applied
    """
    apply_info(model.model.vision_tower.vision_tower, dominant_num=dominant-1, contextual_num=contextual, cluster_width=cluster_width)

    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
        # Ensure all code paths (including pdrop) use the VisionZip-safe encoder
        LlavaMetaForCausalLM.encode_images = encode_images_visionzip

        # Additionally bind on the instance to avoid any MRO/import aliasing issues
        try:
            model.encode_images = types.MethodType(encode_images_visionzip, model)
        except Exception:
            pass

    return model


def visionzip_video(model, dominant=191, contextual=30, cluster_width=4):
    """
    Apply VisionZip to Video-LLaVA models.
    
    This function patches both image and video towers (LanguageBind encoders)
    for Video-LLaVA models that support both images and videos.
    
    Args:
        model: Video-LLaVA model instance
        dominant: Number of dominant (high-attention) tokens to keep
        contextual: Number of contextual (clustered) tokens to keep
        cluster_width: Multiplier for preselection pool size
    
    Returns:
        model: Modified model with VisionZip applied
    """
    from .videollava_arch import (
        encode_images_visionzip_video,
        encode_videos_visionzip,
        restore_image_features_sorted_video,
        prepare_inputs_labels_for_multimodal_visionzip_video,
    )
    
    # Apply VisionZip info to both video and image towers
    apply_info(model.model.video_tower.video_tower, dominant_num=dominant-1, contextual_num=contextual, cluster_width=cluster_width)
    apply_info(model.model.image_tower.image_tower, dominant_num=dominant-1, contextual_num=contextual, cluster_width=cluster_width)

    # Ensure a unified attribute name for inner towers so VisionZip forward can work across towers
    try:
        if hasattr(model.model.image_tower, "image_tower") and not hasattr(model.model.image_tower, "vision_tower"):
            model.model.image_tower.vision_tower = model.model.image_tower.image_tower
    except Exception:
        pass
    try:
        if hasattr(model.model.video_tower, "video_tower") and not hasattr(model.model.video_tower, "vision_tower"):
            model.model.video_tower.vision_tower = model.model.video_tower.video_tower
    except Exception:
        pass

    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    # Patch attention and encoder layers (transformers CLIP as well as LanguageBind CLIP)
    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward
    
    # Patch LanguageBind's CLIP encoder layers to expose metric/_info
    try:
        from videollava.model.multimodal_encoder.languagebind.image.modeling_image import (
            CLIPEncoderLayer as LBImage_CLIPEncoderLayer,
        )
        LBImage_CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    except Exception:
        pass
    try:
        from videollava.model.multimodal_encoder.languagebind.video.modeling_video import (
            CLIPEncoderLayer as LBVideo_CLIPEncoderLayer,
        )
        LBVideo_CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    except Exception:
        pass

    from videollava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward
    
    # Also patch LanguageBind towers to use the same VisionZip forward
    try:
        from videollava.model.multimodal_encoder.languagebind import (
            LanguageBindImageTower,
            LanguageBindVideoTower,
        )
        LanguageBindImageTower.forward = CLIPVisionTower_VisionZip.forward
        LanguageBindVideoTower.forward = CLIPVisionTower_VisionZip.forward
    except Exception:
        pass

    from videollava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip_video
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted_video
        LlavaMetaForCausalLM.encode_videos_visionzip = encode_videos_visionzip
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip_video
        # Ensure all code paths (including pdrop) use the VisionZip-safe encoder
        LlavaMetaForCausalLM.encode_images = encode_images_visionzip_video
        LlavaMetaForCausalLM.encode_videos = encode_videos_visionzip

        # Additionally bind on the instance to avoid any MRO/import aliasing issues
        try:
            model.encode_images = types.MethodType(encode_images_visionzip_video, model)
            model.encode_videos = types.MethodType(encode_videos_visionzip, model)
        except Exception:
            pass

    return model

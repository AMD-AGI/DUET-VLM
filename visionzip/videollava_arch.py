#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Senqiao Yang
# ------------------------------------------------------------------------
# Adapted for DUET-VLM unified codebase
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from videollava.mm_utils import get_anyres_image_grid_shape
import torch.nn.functional as F


def encode_images_visionzip_video(self, images):
    """Encode images using VisionZip for Video-LLaVA models."""
    out = self.get_model().get_image_tower()(images)
    # Support towers that return (features, keep_idx, ...) and those that return features only
    if isinstance(out, (tuple, list)):
        image_features = out[0]
    else:
        image_features = out
    image_features = self.get_model().mm_projector(image_features)
    return image_features


def encode_videos_visionzip(self, videos):  # [mini_b, c, t, h, w]
    """Encode videos using VisionZip for Video-LLaVA models."""
    b, _, t, _, _ = videos.shape
    out = self.get_model().get_video_tower()(videos)  # [mini_b, t, n, c] or tuple

    if isinstance(out, (tuple, list)):
        video_features = out[0]
    else:
        video_features = out

    video_features = self.get_model().mm_projector(video_features)
    return video_features


def restore_image_features_sorted_video(self, image_feature, cur_keep_idx, width, height):
    """Restore image features to their original positions for Video-LLaVA."""
    num_img, total_patches, feature_dim = image_feature.shape
    num_keep = cur_keep_idx.shape[1]  
    num_extra = total_patches - num_keep  

    cur_keep_idx_sorted, _ = cur_keep_idx.sort(dim=1)  # [num_img, num_keep]
    cur_keep_idx_sorted_restore = cur_keep_idx_sorted[:, 1:]-1

    restored_features = torch.zeros((num_img, 576, feature_dim), device=image_feature.device, dtype=image_feature.dtype)  # [num_img, total_patches, feature_dim]

    mask = torch.zeros(num_img, 576, dtype=torch.bool, device=image_feature.device)

    kept_features = image_feature[:, 1:num_keep, :]
    max_patch_slots = restored_features.shape[1]

    for img_idx in range(num_img):
        keep_indices = cur_keep_idx_sorted_restore[img_idx]
        valid_positions = (keep_indices >= 0) & (keep_indices < max_patch_slots)
        keep_indices = keep_indices[valid_positions]

        row_features = kept_features[img_idx]

        if keep_indices.numel() == 0 or row_features.shape[0] == 0:
            continue

        if keep_indices.numel() != row_features.shape[0]:
            min_len = min(keep_indices.numel(), row_features.shape[0])
            keep_indices = keep_indices[:min_len]
            row_features = row_features[:min_len]

        restored_features[img_idx].index_copy_(0, keep_indices, row_features)
        mask[img_idx, keep_indices] = True
    

    assert width * height == restored_features.shape[0], "width * height must equal num_img"
    restored_features = restored_features.view(height, width, 24, 24, feature_dim)  # [height, width, 24, 24, feature_dim]
    restored_features = restored_features.permute(0, 2, 1, 3, 4).contiguous()  # [height, 24, width, 24, feature_dim]
    restored_features = restored_features.view(height, 24, width * 24, feature_dim)  # [height, 24, width*24, feature_dim]
    restored_features = restored_features.view(height * 24, width * 24, feature_dim)  # [height*24, width*24, feature_dim]
    image_newline_expanded = self.model.image_newline.view(1, 1, feature_dim).expand(height * 24, 1, feature_dim).to(restored_features.device)  # [height*24, 1, feature_dim]
    grid_with_newline = restored_features

    mask = mask.view(height, width, 24, 24)  # [height, width, 24, 24]
    mask = mask.permute(0, 2, 1, 3).contiguous()  # [height, 24, width, 24]
    mask = mask.view(height * 24, width * 24)  # [height*24, width*24]

    mask_all = mask

    image_feature_select = grid_with_newline[mask_all]
    raw_img_feature_merge = image_feature[:,-num_extra:,].reshape(-1, feature_dim)
    cls_img_feature_merge = image_feature[:,0,]

    image_feature_select = torch.cat([image_feature_select, cls_img_feature_merge, raw_img_feature_merge])
    return image_feature_select

    
def prepare_inputs_labels_for_multimodal_visionzip_video(
    self, input_ids, position_ids, attention_mask, past_key_values, labels, images
):
    """
    Prepare inputs and labels for multimodal forward pass with VisionZip for Video-LLaVA.
    
    This function handles both images and videos:
    - Images: [3, 224, 224]
    - Videos: [t, 3, 224, 224] - will be flattened to t image features
    """
    # ====================================================================================================
    image_tower = self.get_image_tower()
    video_tower = self.get_video_tower()
    if (image_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
        if past_key_values is not None and (image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
            target_shape = past_key_values[-1][-1].shape[-2] + 1
            attention_mask = torch.cat((attention_mask, torch.ones(
                (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    '''
        images is a list, if batch_size=6
        [
            image(3, 224, 224),      # sample 1
            image(3, 224, 224),      # sample 2
            video(t, 3, 224, 224),   # sample 3
            image(3, 224, 224),      # sample 4
            image(3, 224, 224),      # sample 4
            video(t, 3, 224, 224),   # sample 5
            video(t, 3, 224, 224),   # sample 5
            video(t, 3, 224, 224),   # sample 6
            image(3, 224, 224),      # sample 6
        ]
        will be converted to image_features, all video_feature will be flatten as image
        [
            [n, c],                  # sample 1
            [n, c),                  # sample 2
            *(t * [new_n, c]),       # sample 3
            [n, c],                  # sample 4
            [n, c],                  # sample 4
            *(t * [new_n, c]),       # sample 5
            *(t * [new_n, c]),       # sample 5
            *(t * [new_n, c]),       # sample 6
            [n, c],                  # sample 6
        ]
    '''
    image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
    is_all_image = len(image_idx) == len(images)
    video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 4]
    images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []  # [mini_b, c, h, w]

    # Normalize videos to [c, t, h, w] then stack -> [mini_b, c, t, h, w]
    def _to_cthw(v):
        # If shape is [t, c, h, w] (common), permute to [c, t, h, w]
        if v.shape[1] in (1, 3):
            return v.permute(1, 0, 2, 3)
        return v

    video_tensors = [_to_cthw(images[idx]) for idx in video_idx]
    try:
        videos_minibatch = torch.stack(video_tensors) if len(video_tensors) > 0 else []  # [mini_b, c, t, h, w]
    except Exception:
        # Fallback: will process per-video below
        videos_minibatch = []

    tmp_image_features = [None] * (len(image_idx) + len(video_idx))
    if getattr(images_minibatch, 'ndim', 0) == 4:  # batch consists of images, [mini_b, c, h, w]
        if image_tower is not None:
            image_features_minibatch = self.encode_images_visionzip(images_minibatch)  # [mini_b, l, c]
        else:
            image_features_minibatch = torch.randn(1).to(self.device)  # dummy feature for video-only training under tuning
        for i, pos in enumerate(image_idx):
            tmp_image_features[pos] = image_features_minibatch[i]

    # Encode videos (batch when possible), fallback to per-sample to avoid index errors
    video_features_minibatch = None
    if getattr(videos_minibatch, 'ndim', 0) == 5:  # [mini_b, c, t, h, w]
        video_features_minibatch = self.encode_videos_visionzip(videos_minibatch)  # [mini_b, t, l, c]

    num_encoded = int(getattr(video_features_minibatch, 'shape', [0])[0]) if video_features_minibatch is not None else 0
    for i, pos in enumerate(video_idx):
        if i < num_encoded:
            vf = video_features_minibatch[i]
        else:
            # Per-video fallback
            v_single = _to_cthw(images[pos]).unsqueeze(0)  # [1, c, t, h, w]
            vf = self.encode_videos_visionzip(v_single)[0]  # [t, l, c]
        t = vf.shape[0]
        tmp_image_features[pos] = [vf[j] for j in range(t)]

    new_tmp = []
    for image in tmp_image_features:
        if isinstance(image, list):
            t = len(image)
            for i in range(t):
                new_tmp.append(image[i])
        else:
            new_tmp.append(image)
    image_features = new_tmp
    # ====================================================================================================

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

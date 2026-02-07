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
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder

from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward



class CLIPVisionTower_VisionZip(nn.Module):


    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2]
            metric = self.vision_tower.vision_model.encoder.layers[-2].metric
            dominant_num =  self.vision_tower._info["dominant"]
            contextual_num = self.vision_tower._info["contextual"]
            cluster_width = self.vision_tower._info["cluster_width"]

            ## Dominant Visual Tokens
            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
            cls_attention_sum = cls_attention.sum(dim=1)  
            topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
            # VV: softmax over selected dominant tokens (exclude CLS). Indices for cls_attention_sum are 0-based over patches
            dom_scores = cls_attention_sum.gather(1, topk_indices - 1)
            vv_probs = torch.softmax(dom_scores, dim=1)
            all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)
            
            mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
            dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
            
            ### Filter
            metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])

            hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])  
            
            metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

            ## Contextual Visual Tokens
            # Preselect 4x contextual tokens by top attention (like dominant), then cluster only these
            B, M, D = metric_normalized.shape
            mask_excluding_cls = mask[:, 1:]
            cls_attention_sum_filtered = cls_attention_sum.masked_select(mask_excluding_cls).view(B, M)

            preselect_num = min(max(1, cluster_width * contextual_num), M)
            preselect_indices = cls_attention_sum_filtered.topk(preselect_num, dim=1).indices  # [B, preselect_num]

            # Choose target anchors evenly within the preselected pool
            step = max(1, preselect_num // max(1, contextual_num))
            anchor_pos = torch.arange(0, preselect_num, step, device=metric_normalized.device)[:contextual_num]
            if anchor_pos.numel() == 0:
                # Fallback: keep first token if no anchors (degenerate case)
                anchor_pos = torch.tensor([0], device=metric_normalized.device)
            anchor_pos = anchor_pos.unsqueeze(0).expand(B, -1)

            target_indices = preselect_indices.gather(1, anchor_pos)  # indices in filtered space, shape [B, contextual_num]

            # Build tokens to merge = preselected minus targets
            candidate_keep_mask = torch.ones(preselect_num, dtype=torch.bool, device=metric_normalized.device)
            candidate_keep_mask[anchor_pos[0]] = False  # same anchor positions across batch
            merge_indices = preselect_indices[:, candidate_keep_mask]  # [B, preselect_num - contextual_num]

            # Gather tensors
            expand_dim = lambda idx, dim: idx.unsqueeze(-1).expand(-1, -1, dim)
            target_tokens = torch.gather(metric_normalized, 1, expand_dim(target_indices, D))  # [B, contextual_num, D]

            if merge_indices.shape[1] > 0:
                tokens_to_merge = torch.gather(metric_normalized, 1, expand_dim(merge_indices, D))  # [B, Lm, D]
                similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
                assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], target_tokens.shape[1], dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
                assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                hidden_to_merge = torch.gather(hidden_states_filtered, 1, expand_dim(merge_indices, hidden_states_filtered.shape[2]))
                aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
            else:
                aggregated_hidden = torch.zeros(B, target_tokens.shape[1], hidden_states_filtered.shape[2], dtype=hidden_states_filtered.dtype, device=hidden_states_filtered.device)

            target_hidden = torch.gather(hidden_states_filtered, 1, expand_dim(target_indices, hidden_states_filtered.shape[2]))
            contextual_tokens = target_hidden + aggregated_hidden

            # # Optional: K-means comparison (debug only)
            # debug_kmeans = getattr(self.vision_tower, "debug_cluster", False)
            # if debug_kmeans:
            #     B, N, D = metric_normalized.shape
            #     K = min(contextual_num, N)
            #     I = 3  # Lloyd iterations
            #     init_idx = torch.linspace(0, N - 1, K, device=metric_normalized.device).long()
            #     centers = metric_normalized[:, init_idx, :].float()
            #     for _ in range(I):
            #         sim_k = torch.bmm(metric_normalized.float(), centers.transpose(1, 2))
            #         labels_k = sim_k.argmax(dim=2)  # [B, N]
            #         new_centers = []
            #         for b in range(B):
            #             cs = []
            #             for k in range(K):
            #                 mk = labels_k[b] == k
            #                 if mk.any():
            #                     cs.append(metric_normalized[b][mk].mean(dim=0))
            #                 else:
            #                     cs.append(centers[b, k])
            #             cs = torch.stack(cs, dim=0)
            #             cs = torch.nn.functional.normalize(cs, dim=-1)
            #             new_centers.append(cs)
            #         centers = torch.stack(new_centers, dim=0)
            #     # Store debug info (detach to save memory)
            #     self.vision_tower._debug_clusters = {
            #         "baseline_target_indices": target_indices.detach().cpu(),
            #         "baseline_assign_argmax": similarity.argmax(dim=2).detach().cpu(),
            #         "kmeans_labels": labels_k.detach().cpu(),
            #     }

            # Merge with target hidden states and concatenate
            hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)

        return hidden_states_save, all_indices#, vv_probs

        # return hidden_states_save, hidden_states, all_indices






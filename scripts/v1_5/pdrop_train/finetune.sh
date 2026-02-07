#!/bin/bash

# export layer_list="[16,24]"
# export image_token_ratio_list="[0.5,0.0]"

export RUN_NAME="${1:-VZPD-finetune-cw4-trial-ALL}"
export OUTPUT_DIR="${2:-/wekafs/aditysin/checkpoints/llava-v1.5-7b-finetune-vzpd-cw4-trial-ALL}"
export layer_list="${3:-[16,24]}"
export image_token_ratio_list="${4:-[0.5,0.0]}"
export PRETRAIN_MM_MLP_ADAPTER="${5:-/wekafs/aditysin/checkpoints/llava-v1.5-7b-pretrain-vzpd-cw4-trial-ALL/mm_projector.bin}"
export dominant_num="${6:-300}"
export context_num="${7:-7}"
export cluster_width="${8:-4}"
export WANDB="${9:-wandb}"

echo "Running Finetune with the following parameters:"
echo "RUN_NAME: ${RUN_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "layer_list: ${layer_list}"
echo "image_token_ratio_list: ${image_token_ratio_list}"
echo "PRETRAIN_MM_MLP_ADAPTER: ${PRETRAIN_MM_MLP_ADAPTER}"
echo "dominant_num: ${dominant_num}"
echo "context_num: ${context_num}"
echo "cluster_width: ${cluster_width}"
echo "WANDB: ${WANDB}"

# exit 0

deepspeed llava/train/train_mem_pdrop.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /wekafs/aditysin/LLaVA/llava_v1_5_mix665k.json \
    --image_folder /wekafs/aditysin/LLaVA \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --dominant_num ${dominant_num} \
    --context_num ${context_num} \
    --cluster_width ${cluster_width} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --layer_list  ${layer_list} \
    --image_token_ratio_list ${image_token_ratio_list} \
    --report_to ${WANDB}

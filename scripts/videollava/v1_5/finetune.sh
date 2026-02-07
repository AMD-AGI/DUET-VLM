#!/bin/bash
export PYTHONPATH=/home/hkandala@amd.com/code/faster_VLM/Video-LLaVA
export WANDB_API_KEY=REDACTED_WANDB_API_KEY

JSON_FOLDER="/wekafs/hkandala/video_data/train_json"
IMAGE_FOLDER="/wekafs/hkandala/LLaVA_video_data/llava_all_image_video"
VIDEO_FOLDER="/wekafs/hkandala/video_llava_training_data"

deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /wekafs/hkandala/LLaVA_data/llava_v1_5_mix665k.json ${JSON_FOLDER}/videochatgpt_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder /wekafs/hkandala/LLaVA_data \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /wekafs/hkandala/checkpoints/videollava-7b-pretrain-vzpd-pdrop-compare/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --monitor_grads True \
    --monitor_grads_every 1 \
    --raise_on_nan_grad True \
    --output_dir /wekafs/hkandala/checkpoints/videollava-7b-vzpd-pdrop-compare-test \
    --run_name VZPD-finetune-pdrop-compare-test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --dominant_num 160 \
    --context_num 32 \
    --layer_list "[16,24]" \
    --image_token_ratio_list "[0.5,0.0]" \
    --warmup_ratio 0.03  \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --cache_dir "/wekafs/hkandala/cache_dir"

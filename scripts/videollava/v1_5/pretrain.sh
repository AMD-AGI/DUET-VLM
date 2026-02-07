#!/bin/bash
export PYTHONPATH=/home/hkandala@amd.com/code/faster_VLM/Video-LLaVA
export WANDB_API_KEY=REDACTED_WANDB_API_KEY

JSON_FOLDER="/wekafs/hkandala/video_data/train_json"
IMAGE_FOLDER="/wekafs/hkandala/LLaVA_video_data/llava_all_image_video"
VIDEO_FOLDER="/wekafs/hkandala/video_llava_training_data"
export layer_list="[16,24]"
export image_token_ratio_list="[0.5,0.0]"
# cd /wekafs/hkandala/Video-LLaVA
deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /wekafs/hkandala/LLaVA_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json ${JSON_FOLDER}/valley_.json \
    --image_folder /wekafs/hkandala/LLaVA_data/LLaVA-Pretrain/images \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /wekafs/hkandala/checkpoints/videollava-7b-pretrain-vzpd-pdrop-compare \
    --run_name videollava-pretrain-vzpd-pdrop-compare \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --dominant_num 160 \
    --context_num 32 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --layer_list  ${layer_list} \
    --image_token_ratio_list ${image_token_ratio_list} \
    --report_to wandb \
    --cache_dir "/wekafs/hkandala/cache_dir"
#!/usr/bin/env bash
set -euo pipefail

# Runs the Video-LLaVA finetuning job across 8 GPUs inside a detached tmux session.
# Usage:
#   ./run_videollava_finetune_tmux_8gpu.sh [session-name]
#
# Optionally export CUDA_VISIBLE_DEVICES before invoking to choose specific GPUs
# (must resolve to exactly 8 device IDs). The script defaults to 0-7 otherwise.
# WANDB_API_KEY should be set in the environment if wandb logging is desired.

SESSION_NAME="${1:-videollava_finetune_8gpu}"
PROJECT_ROOT="/home/hkandala@amd.com/code/faster_VLM/Video-LLaVA"
PYTHON_BIN="/opt/conda/envs/py_3.10/bin/python"
MASTER_PORT="${MASTER_PORT:-29518}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed or not in PATH." >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Error: tmux session \"${SESSION_NAME}\" already exists." >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

IFS=',' read -ra GPU_IDS <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_IDS[@]}"
if (( NUM_GPUS != 8 )); then
    echo "Error: CUDA_VISIBLE_DEVICES must reference exactly 8 GPUs (currently \"${GPU_LIST}\")." >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Warning: WANDB_API_KEY is not set; wandb logging will fail." >&2
fi

CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node "${NUM_GPUS}"
    --master_port "${MASTER_PORT}"
    --module videollava.train.train_mem
    --deepspeed "${PROJECT_ROOT}/scripts/zero2.json"
    --model_name_or_path "lmsys/vicuna-7b-v1.5"
    --version "v1"
    --data_path "/wekafs/hkandala/LLaVA_data/llava_v1_5_mix665k.json" "/wekafs/hkandala/video_data/train_json/videochatgpt_tune_.json" "/wekafs/hkandala/video_data/train_json/nlp_tune.json"
    --image_folder "/wekafs/hkandala/LLaVA_data"
    --image_tower "LanguageBind/LanguageBind_Image"
    --video_folder "/wekafs/hkandala/video_llava_training_data"
    --video_tower "LanguageBind/LanguageBind_Video_merge"
    --mm_projector_type "mlp2x_gelu"
    --pretrain_mm_mlp_adapter "/wekafs/hkandala/checkpoints/videollava-7b-pretrain-vzpd-pdrop-compare/mm_projector.bin"
    --mm_vision_select_layer "-2"
    --mm_use_im_start_end "False"
    --mm_use_im_patch_token "False"
    --image_aspect_ratio "pad"
    --group_by_modality_length "True"
    --bf16 "True"
    --output_dir "/wekafs/hkandala/checkpoints/videollava-7b-vzpd-pdrop-compare"
    --run_name "VZPD-finetune-pdrop-compare-8gpu"
    --num_train_epochs "1"
    --per_device_train_batch_size "16"
    --per_device_eval_batch_size "4"
    --gradient_accumulation_steps "1"
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps "1000"
    --save_total_limit "1"
    --learning_rate "2e-5"
    --weight_decay "0."
    --warmup_ratio "0.03"
    --lr_scheduler_type "cosine"
    --logging_steps "1"
    --tf32 "False"
    --monitor_grads "True"
    --monitor_grads_every "100"
    --raise_on_nan_grad "True"
    --layer_list "[16,24]"
    --image_token_ratio_list "[0.5,0.0]"
    --dominant_num "160"
    --context_num "32"
    --model_max_length "2048"
    --tokenizer_model_max_length "3072"
    --gradient_checkpointing "True"
    --dataloader_num_workers "8"
    --lazy_preprocess "True"
    --report_to "wandb"
    --cache_dir "/wekafs/hkandala/cache_dir"
)

CMD_STRING="$(printf '%q ' "${CMD[@]}")"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd \"${PROJECT_ROOT}\" && export PYTHONPATH=\"${PROJECT_ROOT}\" && export CUDA_VISIBLE_DEVICES=\"${GPU_LIST}\" && exec ${CMD_STRING}"

echo "Started tmux session \"${SESSION_NAME}\" on GPUs ${GPU_LIST}."
echo "Attach with: tmux attach -t ${SESSION_NAME}"



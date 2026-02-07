#!/bin/bash

CUSTOM_NAME="${1:-vzpd_cw4_192}"
MODEL_PATH="${2:-liuhaotian/llava-v1.5-7b}"
EXTRA_ARGS="${3:---layer_list [16,24] --image_token_ratio_list [0.5,0.0] --dominant 300 --contextual 7 --cluster_width 4 --conv_mode vicuna_v1 --compute_salient_tokens True}"

# ----------------------------------------
# Parse known args from EXTRA_ARGS
# ----------------------------------------
# Use regex + grep/sed to extract values
get_arg() {
  local key="$1"
  echo "$EXTRA_ARGS" | grep -oP "(?<=--$key\s)[^ ]+"
}

LAYER_LIST=$(get_arg layer_list)
RATIO_LIST=$(get_arg image_token_ratio_list)
DOMINANT=$(get_arg dominant)
CONTEXTUAL=$(get_arg contextual)
CLUSTER_WIDTH=$(get_arg cluster_width)
CONV_MODE=$(get_arg conv_mode)
COMPUTE_SALIENT_TOKENS=$(get_arg compute_salient_tokens)
echo "Parsed values:"
echo "  CUSTOM_NAME = $CUSTOM_NAME"
echo "  MODEL_PATH = $MODEL_PATH"
echo "  LAYER_LIST = $LAYER_LIST"
echo "  RATIO_LIST = $RATIO_LIST"
echo "  DOMINANT   = $DOMINANT"
echo "  CONTEXTUAL = $CONTEXTUAL"
echo "  CLUSTER_WIDTH = $CLUSTER_WIDTH"
echo "  CONV_MODE = $CONV_MODE"
echo "  COMPUTE_SALIENT_TOKENS = $COMPUTE_SALIENT_TOKENS"
# ----------------------------------------
# GPU configuration
# ----------------------------------------

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"
VQAV2_DIR="/wekafs/aditysin/PyramidDrop/data/playground/data/eval/vqav2"

# ----------------------------------------
# Launch model with final resolved values
# ----------------------------------------

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $VQAV2_DIR/$SPLIT.jsonl \
        --image-folder $VQAV2_DIR/test2015 \
        --answers-file $VQAV2_DIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_${CUSTOM_NAME}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --layer_list  $LAYER_LIST \
        --image_token_ratio_list $RATIO_LIST \
        --dominant $DOMINANT \
        --contextual $CONTEXTUAL \
        --cluster_width $CLUSTER_WIDTH \
        ${COMPUTE_SALIENT_TOKENS:+--compute_salient_tokens} \
        --conv-mode $CONV_MODE &
done

wait

output_file=$VQAV2_DIR/answers/$SPLIT/$CKPT/merge_${CUSTOM_NAME}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $VQAV2_DIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_${CUSTOM_NAME}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --dir $VQAV2_DIR --split $SPLIT --ckpt $CKPT --custom_name $CUSTOM_NAME


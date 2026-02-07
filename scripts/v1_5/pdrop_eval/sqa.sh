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
# Launch model with final resolved values
# ----------------------------------------

python -m llava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /wekafs/aditysin/PyramidDrop/data/playground/data/eval/scienceqa/images/test \
    --answers-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/scienceqa/answers/llava-v1.5-7b_${CUSTOM_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list  $LAYER_LIST \
    --image_token_ratio_list $RATIO_LIST \
    --dominant $DOMINANT \
    --contextual $CONTEXTUAL \
    --cluster_width $CLUSTER_WIDTH \
    ${COMPUTE_SALIENT_TOKENS:+--compute_salient_tokens} \
    --conv-mode $CONV_MODE
    # --model-path /wekafs/hkandala/checkpoints/vzpd/llava-v1.5-7b-finetune-64-cw-16 \



python llava/eval/eval_science_qa.py \
    --base-dir /wekafs/aditysin/PyramidDrop/data/playground/data/eval/scienceqa \
    --result-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/scienceqa/answers/llava-v1.5-7b_${CUSTOM_NAME}.jsonl \
    --output-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/scienceqa/answers/llava-v1.5-7b_output_${CUSTOM_NAME}.jsonl \
    --output-result /wekafs/aditysin/PyramidDrop/data/playground/data/eval/scienceqa/answers/llava-v1.5-7b_result_${CUSTOM_NAME}.json

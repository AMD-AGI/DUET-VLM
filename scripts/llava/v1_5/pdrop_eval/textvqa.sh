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
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /wekafs/aditysin/PyramidDrop/data/playground/data/eval/textvqa/train_images \
    --answers-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/textvqa/answers/llava-v1.5-7b-finetune-192_${CUSTOM_NAME}.jsonl \
    --temperature 0 \
    --layer_list  $LAYER_LIST \
    --image_token_ratio_list $RATIO_LIST \
    --dominant $DOMINANT \
    --contextual $CONTEXTUAL \
    --cluster_width $CLUSTER_WIDTH \
    ${COMPUTE_SALIENT_TOKENS:+--compute_salient_tokens} \
    --conv-mode $CONV_MODE
    # --model-path liuhaotian/llava-v1.5-7b \

python -m llava.eval.eval_textvqa \
    --annotation-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /wekafs/aditysin/PyramidDrop/data/playground/data/eval/textvqa/answers/llava-v1.5-7b-finetune-192_${CUSTOM_NAME}.jsonl

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

# ----------------------------------------
# Launch model with final resolved values
# ----------------------------------------

CKPT="llava-v1.5-7b"
SEED_BENCH_DIR="/wekafs/aditysin/PyramidDrop/data/playground/data/eval/seed_bench"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $SEED_BENCH_DIR/llava-seed-bench.jsonl \
        --image-folder $SEED_BENCH_DIR \
        --answers-file $SEED_BENCH_DIR/answers/$CKPT/${CHUNKS}_${IDX}_${CUSTOM_NAME}.jsonl \
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
# ============= MODS =============
mkdir -p "$SEED_BENCH_DIR/answers/$CKPT"
mkdir -p "$SEED_BENCH_DIR/answers_upload"
# ============= MODS =============

output_file=$SEED_BENCH_DIR/answers/$CKPT/merge_${CUSTOM_NAME}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SEED_BENCH_DIR/answers/$CKPT/${CHUNKS}_${IDX}_${CUSTOM_NAME}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file $SEED_BENCH_DIR/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $SEED_BENCH_DIR/answers_upload/$CKPT_${CUSTOM_NAME}.jsonl


# # Information on how to prepare SEED-Bench-video-image folder/dir:

# Steps:
# 1. `mkdir -p playground/data/eval/seed_bench/SEED-Bench-video-image`
# 2. `cd playground/data/eval/seed_bench/SEED-Bench-video-image`
# 3. `mkdir 10 11 12`
# 4. `cd 10`
# 5. `wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01 && wget https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00 && wget https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip`
# 6. `cat 20bn-something-something-v2-?? >> 20bn-something-something-v2.tar.gz && tar -xzvf 20bn-something-something-v2.tar.gz`
# 7. `cd ..`  
# 8. `cd 11`
# 9. `git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git`
# 10. `cd epic-kitchens-download-scripts`
# 11. `python epic_downloader.py --output-path /wekafs/aditysin/PyramidDrop/data/playground/data/eval/seed_bench/SEED-Bench-video-image/11 --val --videos --rgb-frames` <!-- Run this command to download the validation set and ensure your python>3.5 -->
# 12. `cd ..`
# 13. `cd 12`
# 14. `pip install -U gdown && gdown https://drive.google.com/uc?id=1jgSoof1AatiDRpGY091qd4TEKF-BUt6I && tar -xzvf BreakfastII_15fps_qvga_sync.tar.gz` <!-- Run this command to download the BreakfastII dataset -->
# 15. `cd ..`
# 16. `cat v1_video.zip.??? >> v1_video.zip && unzip v1_video.zip` <!-- download v1_video.zip.??? from HF and cat & unzip -->
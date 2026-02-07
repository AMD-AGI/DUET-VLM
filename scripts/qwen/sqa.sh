#!/bin/bash
# Qwen2.5-VL DUET - ScienceQA Evaluation
# Usage: bash scripts/qwen/sqa.sh [CUSTOM_NAME] [MODEL_PATH] [EXTRA_ARGS]
#
# Examples:
#   bash scripts/qwen/sqa.sh duet_640                          # DUET defaults
#   bash scripts/qwen/sqa.sh vz_orig_640 Qwen/Qwen2.5-VL-7B-Instruct "--mode ori_visionzip --dominant 540 --contextual 100"
#   bash scripts/qwen/sqa.sh baseline Qwen/Qwen2.5-VL-7B-Instruct "--mode baseline"

CUSTOM_NAME="${1:-duet_640}"
MODEL_PATH="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXTRA_ARGS="${3:---mode duet --dominant 540 --contextual 100 --layer-list 14 21 --ratio-list 0.5 0.25}"

DATA_ROOT="/wekafs/aditysin/PyramidDrop/data/playground/data/eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../qwen2_5_vl" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Qwen2.5-VL DUET - ScienceQA Evaluation"
echo "  CUSTOM_NAME: $CUSTOM_NAME"
echo "  MODEL_PATH:  $MODEL_PATH"
echo "  EXTRA_ARGS:  $EXTRA_ARGS"
echo "============================================================"

# Step 1: Run inference + evaluation
python "${SCRIPT_DIR}/eval_benchmarks.py" \
    --model-path "$MODEL_PATH" \
    --benchmark sqa \
    --question-file "${DATA_ROOT}/scienceqa/llava_test_CQM-A.json" \
    --image-folder "${DATA_ROOT}/scienceqa/images/test" \
    --output-file "${RESULTS_DIR}/sqa_${CUSTOM_NAME}.jsonl" \
    --warmup 1 \
    $EXTRA_ARGS

echo ""
echo "Results saved to: ${RESULTS_DIR}/sqa_${CUSTOM_NAME}.jsonl"

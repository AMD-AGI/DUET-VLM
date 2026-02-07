#!/bin/bash
# Qwen2.5-VL DUET - GQA Evaluation
# Usage: bash scripts/qwen/gqa.sh [CUSTOM_NAME] [MODEL_PATH] [EXTRA_ARGS]
#
# Examples:
#   bash scripts/qwen/gqa.sh duet_640                          # DUET defaults
#   bash scripts/qwen/gqa.sh vz_orig_640 Qwen/Qwen2.5-VL-7B-Instruct "--mode ori_visionzip --dominant 540 --contextual 100"
#   bash scripts/qwen/gqa.sh baseline Qwen/Qwen2.5-VL-7B-Instruct "--mode baseline"

CUSTOM_NAME="${1:-duet_640}"
MODEL_PATH="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXTRA_ARGS="${3:---mode duet --dominant 540 --contextual 100 --layer-list 14 21 --ratio-list 0.5 0.25}"

DATA_ROOT="/wekafs/aditysin/PyramidDrop/data/playground/data/eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../qwen2_5_vl" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
GQADIR="${DATA_ROOT}/gqa"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Qwen2.5-VL DUET - GQA Evaluation"
echo "  CUSTOM_NAME: $CUSTOM_NAME"
echo "  MODEL_PATH:  $MODEL_PATH"
echo "  EXTRA_ARGS:  $EXTRA_ARGS"
echo "============================================================"

# Step 1: Run inference
python "${SCRIPT_DIR}/eval_benchmarks.py" \
    --model-path "$MODEL_PATH" \
    --benchmark gqa \
    --question-file "${GQADIR}/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "${GQADIR}/data/images" \
    --output-file "${RESULTS_DIR}/gqa_${CUSTOM_NAME}.jsonl" \
    --warmup 1 \
    $EXTRA_ARGS

# Step 2: Convert for official eval
echo ""
echo "Converting results for GQA official eval..."
python scripts/convert_gqa_for_eval.py \
    --src "${RESULTS_DIR}/gqa_${CUSTOM_NAME}.jsonl" \
    --dst "${GQADIR}/testdev_balanced_qwen_${CUSTOM_NAME}_predictions.json"

# Step 3: Run official eval
cd "${GQADIR}/data"
python eval.py --tier testdev_balanced \
    --predictions "${GQADIR}/testdev_balanced_qwen_${CUSTOM_NAME}_predictions.json"

echo ""
echo "Results saved to: ${RESULTS_DIR}/gqa_${CUSTOM_NAME}.jsonl"

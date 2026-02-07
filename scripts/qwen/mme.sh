#!/bin/bash
# Qwen2.5-VL DUET - MME Evaluation
# Usage: bash scripts/qwen/mme.sh [CUSTOM_NAME] [MODEL_PATH] [EXTRA_ARGS]
#
# Examples:
#   bash scripts/qwen/mme.sh duet_640                          # DUET defaults
#   bash scripts/qwen/mme.sh vz_orig_640 Qwen/Qwen2.5-VL-7B-Instruct "--mode ori_visionzip --dominant 540 --contextual 100"
#   bash scripts/qwen/mme.sh baseline Qwen/Qwen2.5-VL-7B-Instruct "--mode baseline"

CUSTOM_NAME="${1:-duet_640}"
MODEL_PATH="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXTRA_ARGS="${3:---mode duet --dominant 540 --contextual 100 --layer-list 14 21 --ratio-list 0.5 0.25}"

DATA_ROOT="/wekafs/aditysin/PyramidDrop/data/playground/data/eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../qwen2_5_vl" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
MMEDIR="${DATA_ROOT}/MME"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Qwen2.5-VL DUET - MME Evaluation"
echo "  CUSTOM_NAME: $CUSTOM_NAME"
echo "  MODEL_PATH:  $MODEL_PATH"
echo "  EXTRA_ARGS:  $EXTRA_ARGS"
echo "============================================================"

# Step 1: Run inference
python "${SCRIPT_DIR}/eval_benchmarks.py" \
    --model-path "$MODEL_PATH" \
    --benchmark mme \
    --question-file "${MMEDIR}/llava_mme.jsonl" \
    --image-folder "${MMEDIR}/MME_Benchmark_release_version" \
    --output-file "${RESULTS_DIR}/mme_${CUSTOM_NAME}.jsonl" \
    --warmup 1 \
    $EXTRA_ARGS

# Step 2: Convert + evaluate with official MME tools
echo ""
echo "Converting results for MME official eval..."
cd "$MMEDIR"
python convert_answer_to_mme.py --experiment "qwen_${CUSTOM_NAME}"

cd eval_tool
python calculation.py --results_dir "answers/qwen_${CUSTOM_NAME}"

echo ""
echo "Results saved to: ${RESULTS_DIR}/mme_${CUSTOM_NAME}.jsonl"

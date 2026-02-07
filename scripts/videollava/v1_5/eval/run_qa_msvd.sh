

CKPT_NAME="Video-LLaVA-7B-vzpd-pdrop-inf"
# model_path="/wekafs/hkandala/checkpoints/videollava-7b-vzpd-pdrop-compare"
model_path="LanguageBind/Video-LLaVA-7B"

# model_path="/wekafs/hkandala/checkpoints/videollava-7b-vzpd-working/checkpoint-5000-freeze"
cache_dir="/wekafs/hkandala/cache_dir"
GPT_Zero_Shot_QA="/wekafs/aditysin/PyramidDrop/data/playground/data/eval/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/videos"
gt_file_question="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/test_a.json"
output_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${CKPT_NAME}"
layer_list="[16,24]"
image_token_ratio_list="[0.5,0.0]"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


# Track background process IDs for clean termination
pids=()

terminate() {
  echo "Termination signal received. Stopping all jobs..."
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  # Wait for children to exit to avoid zombies
  wait "${pids[@]}" 2>/dev/null
  exit 130
}

# Handle Ctrl+C and termination signals
trap terminate INT TERM


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollava/eval/video/run_inference_video_qa.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --layer_list ${layer_list} \
      --image_token_ratio_list ${image_token_ratio_list} \
      --dominant 160 \
      --contextual 32 \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
  pids+=("$!")
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done
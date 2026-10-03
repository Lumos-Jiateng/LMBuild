#!/usr/bin/env bash
# The two Level 3 judges (non-contestants), as served for the paper: one 80 GB GPU each.
#   bash scripts/serve_judges.sh <gpu_qwen> <gpu_gemma>
# then: JUDGE_QWEN_URL=http://127.0.0.1:8111/v1 JUDGE_GEMMA_URL=http://127.0.0.1:8112/v1 bash scripts/evaluate.sh
set -euo pipefail
GQ="${1:-0}"; GG="${2:-1}"
CUDA_VISIBLE_DEVICES=$GQ vllm serve Qwen/Qwen2.5-VL-32B-Instruct --served-model-name qwen2.5-vl-32b --port 8111 \
    --gpu-memory-utilization 0.92 --max-model-len 16384 --max-num-seqs 16 --limit-mm-per-prompt '{"image": 2}' > vllm_judge_qwen.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GG vllm serve google/gemma-3-27b-it --served-model-name gemma-3-27b --port 8112 \
    --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 16 --limit-mm-per-prompt '{"image": 2}' > vllm_judge_gemma.log 2>&1 &
echo "judges starting (logs: vllm_judge_qwen.log, vllm_judge_gemma.log)"
wait

#!/usr/bin/env bash
# Serve one open model from configs/open_models.tsv with vLLM and run it over the core set (main setting by default).
#
#   bash scripts/run_open_model.sh <system_name> [GPU] [PORT] [extra core_batch args...]
#   e.g. bash scripts/run_open_model.sh qwen3.5-27b 0 8101 --tasks office_desk,toilet --tiers B --conditions name_only+image
#
# Needs `vllm` on PATH (the paper used vLLM 0.1x on A100 80 GB; fp8 weights on SM80 need --quantization fp8, and
# fp8 KV cache is refused there). Episodes go to results/v2/<task>/runs/core_v2.4/<system>__<tier>__<cond>__r3__s0/.
set -euo pipefail
cd "$(dirname "$0")/.."
SYS="${1:?system_name from configs/open_models.tsv}"; GPU="${2:-0}"; PORT="${3:-8101}"; shift $(( $# < 3 ? $# : 3 ))
ROW=$(awk -F'\t' -v s="$SYS" '$1==s' configs/open_models.tsv)
[ -z "$ROW" ] && { echo "unknown system $SYS"; exit 2; }
IFS=$'\t' read -r _ REPO SERVED EXTRA NOVIS CONC VARGS <<< "$ROW"
CUDA_VISIBLE_DEVICES=$GPU vllm serve "$REPO" --served-model-name "$SERVED" --port "$PORT" --gpu-memory-utilization 0.90 \
    --max-model-len 32768 --max-num-seqs "$CONC" --limit-mm-per-prompt '{"image": 2}' $VARGS > "vllm_$SERVED.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see vllm_$SERVED.log"; exit 1; }; sleep 10
done
PYTHONPATH=. .venv_eval/bin/python -m ppbench.v2.core_batch --model "$SERVED" --system-name "$SYS" \
    --base-url "http://127.0.0.1:$PORT/v1" --concurrency "$CONC" --extra-body "$EXTRA" $NOVIS "$@"

#!/usr/bin/env bash
set -euo pipefail

# Run all model/dataset/prompt_type evaluations sequentially, then generate the summary
# Usage: ./scripts/run_all_evals_and_summary.sh [--dry-run]

DRY_RUN=false
if [[ ${1:-} == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Defaults
MAX_NEW_TOKENS=10
TEMPERATURE=0.0
SEED=42

# Define models and datasets
# MODELS=("Gemma-3-1B" "Gemma-3-4B" "Gemma-3-27B" "Llama-3.1-8B" "Llama-3.2-1B" "Llama-3.2-3B" "Llama-3.3-70B" "Phi-3")
MODELS=("Llama-3.2-1B" "Llama-3.2-3B" "Llama-3.3-70B" "Phi-3")
DATASETS=("clariq" "ambignq")
PROMPT_TYPES=("zero_shot" "few_shot")

# Model-specific batch sizes and GPU selection (adjust as needed)
# Note: GPU selection is via CUDA_VISIBLE_DEVICES environment variable.
# Set $GPUS before running the script if you want a different default.
DEFAULT_GPUS=${GPUS:-0}

declare -A BATCH_SIZES
BATCH_SIZES["Gemma-3-1B"]=128
BATCH_SIZES["Gemma-3-4B"]=128
BATCH_SIZES["Gemma-3-27B"]=32
BATCH_SIZES["Llama-3.1-8B"]=64
BATCH_SIZES["Llama-3.2-1B"]=128
BATCH_SIZES["Llama-3.2-3B"]=128
BATCH_SIZES["Llama-3.3-70B"]=32
BATCH_SIZES["Phi-3"]=128

# Model-specific GPU counts (set which GPUs to use for that model if needed)
declare -A GPUS_REQUIRED
GPUS_REQUIRED["Gemma-3-27B"]="0,1,2,3"  # example: requires 4 GPUs
GPUS_REQUIRED["Llama-3.3-70B"]="0,1,2,3"  # example: requires 4 GPUs
# Other models will use DEFAULT_GPUS by default (one GPU)

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_SCRIPT="$PROJECT_ROOT/evaluations/llm_inference/evaluate.py"
LOGS_DIR="$PROJECT_ROOT/results/llm"

echo "Project root: $PROJECT_ROOT"

# Run evaluations
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for prompt in "${PROMPT_TYPES[@]}"; do

      batch_size=${BATCH_SIZES[$model]:-32}
      # Use specific GPU list if provided, else use the default
      gpu_list=${GPUS_REQUIRED[$model]:-$DEFAULT_GPUS}

      cmd=(python "$EVAL_SCRIPT"
           --models "$model"
           --datasets "$dataset"
           --prompt_type "$prompt"
           --max_new_tokens "$MAX_NEW_TOKENS"
           --temperature "$TEMPERATURE"
           --batch_size "$batch_size"
           --seed "$SEED")

      echo "\n=== Running: Model=$model Dataset=$dataset Prompt=$prompt BatchSize=$batch_size GPUs=$gpu_list ==="
      if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: CUDA_VISIBLE_DEVICES=$gpu_list ${cmd[*]}"
      else
        echo "Logging: $(date '+%Y%m%d_%H%M%S') -> logs/"
        # Execute with GPU selection
        CUDA_VISIBLE_DEVICES="$gpu_list" "${cmd[@]}"
      fi

    done
  done

done

# Print a friendly finish message
echo "\nAll evaluations complete. If you ran without --dry-run, summary created at: $OUTPUT_SUMMARY"; exit 0

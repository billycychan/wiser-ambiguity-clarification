#!/bin/bash

# Script to run PLM baselines (DistilBERT and RoBERTa) for zero-shot classification
# Usage: ./run_plm_baselines.sh

# Make it executable
# chmod +x intent_cf/scripts/run_plm_baselines.sh

# # Run both baselines
# ./intent_cf/scripts/run_plm_baselines.sh


set -e  # Exit on error

# Change to the baselines directory
cd "$(dirname "$0")/../plm_training/baselines"

echo "=========================================="
echo "Running PLM Baseline Evaluations"
echo "=========================================="
echo ""

# Array of baseline models to run
MODELS=("distillbert" "roberta")

# Run evaluation for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting evaluation with: ${MODEL}"
    echo "=========================================="
    
    python "${MODEL}.py"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Evaluation failed for ${MODEL} with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
    
    echo "Completed: ${MODEL}"
    echo ""
done

echo ""
echo "=========================================="
echo "All baseline evaluations completed!"
echo "=========================================="
echo "Output files saved to:"
echo "  - plm_training/baselines/logs/distillbert/"
echo "  - plm_training/baselines/logs/roberta/"

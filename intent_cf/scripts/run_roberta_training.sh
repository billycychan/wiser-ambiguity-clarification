#!/bin/bash

# Script to train RoBERTa on all synthetic datasets
# Usage: ./run_roberta_training.sh

set -e  # Exit on error

# Change to the plm_training directory
cd "$(dirname "$0")/../plm_training"

echo "=========================================="
echo "Training RoBERTa on All Datasets"
echo "=========================================="
echo ""

# Array of datasets to train on
DATASETS=("llama31_8b" "gemma-3-27b" "llama3.3-70B" "gpt-4-1" "gpt-4-1-mini" "gpt-4-1-nano")

# Train on each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training on dataset: ${DATASET}"
    echo "=========================================="
    
    python roberta.py --dataset "$DATASET"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Training failed for ${DATASET} with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
    
    echo "Completed training on: ${DATASET}"
    echo ""
done

echo ""
echo "=========================================="
echo "All RoBERTa training completed!"
echo "=========================================="
echo "Output files saved to:"
echo "  - ../logs/roberta/"
echo ""
echo "Available datasets trained on:"
for DATASET in "${DATASETS[@]}"; do
    echo "  - ${DATASET}"
done

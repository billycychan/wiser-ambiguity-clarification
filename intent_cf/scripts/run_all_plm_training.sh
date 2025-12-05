#!/bin/bash

# Script to train both DistilBERT and RoBERTa on all synthetic datasets
# Usage: ./run_all_plm_training.sh

set -e  # Exit on error

SCRIPT_DIR="$(dirname "$0")"

echo "=========================================="
echo "Training All PLM Models on All Datasets"
echo "=========================================="
echo ""

# Run DistilBERT training
echo "Starting DistilBERT training..."
bash "${SCRIPT_DIR}/run_distillbert_training.sh"

echo ""
echo "=========================================="
echo ""

# Run RoBERTa training
echo "Starting RoBERTa training..."
bash "${SCRIPT_DIR}/run_roberta_training.sh"

echo ""
echo "=========================================="
echo "All PLM training completed successfully!"
echo "=========================================="
echo "Results saved to:"
echo "  - ../logs/distillbert/"
echo "  - ../logs/roberta/"

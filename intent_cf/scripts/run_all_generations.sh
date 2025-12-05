#!/bin/bash

# Script to run synthetic query generation for all available models
# Usage: ./run_all_generations.sh [batch_size] [limit]
#   batch_size: Optional, default is 32
#   limit: Optional, limits number of topics for testing
# # Make it executable
# chmod +x intent_cf/scripts/run_all_generations.sh

# # Run with default settings (batch_size=32, all topics)
# ./intent_cf/scripts/run_all_generations.sh

# # Run with custom batch size
# ./intent_cf/scripts/run_all_generations.sh 16

# # Run with custom batch size and topic limit (for testing)
# ./intent_cf/scripts/run_all_generations.sh 16 10

set -e  # Exit on error

# Change to the synthgen directory
cd "$(dirname "$0")/../synthgen"

# Parse arguments
BATCH_SIZE=${1:-32}
LIMIT=${2:-}

echo "=========================================="
echo "Running Synthetic Query Generation"
echo "=========================================="
echo "Batch size: $BATCH_SIZE"
if [ -n "$LIMIT" ]; then
    echo "Topic limit: $LIMIT (testing mode)"
else
    echo "Topic limit: None (full generation)"
fi
echo "=========================================="
echo ""

# Array of models to run
MODELS=("Llama-3.1-8B" "Llama-3.3-70B" "Gemma-3-27B")

# Run generation for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting generation with: $MODEL"
    echo "=========================================="
    
    if [ -n "$LIMIT" ]; then
        python generate.py --model "$MODEL" --batch_size "$BATCH_SIZE" --limit "$LIMIT"
    else
        python generate.py --model "$MODEL" --batch_size "$BATCH_SIZE"
    fi
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Generation failed for $MODEL with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
    
    echo "Completed: $MODEL"
    echo ""
done

echo ""
echo "=========================================="
echo "All generations completed successfully!"
echo "=========================================="
echo "Output files saved to: ../data/"
echo "Log files saved to: logs/"

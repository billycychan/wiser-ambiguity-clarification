#!/bin/bash
# Setup script for the 'clarification' conda environment
# This script creates a fresh conda environment with all necessary dependencies
# for running the clarification.py script with PyTorch, CUDA, and vLLM

set -e  # Exit on error

ENV_NAME="clarification"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "Setting up ${ENV_NAME} environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment
echo "Creating new conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate the environment
echo "Activating ${ENV_NAME} environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (this will automatically install transformers, huggingface-hub, etc.)
echo "Installing vLLM (includes transformers, tokenizers, and other dependencies)..."
pip install vllm>=0.11.2

# Install critical dependencies
echo "Installing critical fixes..."
pip install brotlicffi==1.2.0.0 gmpy2==2.2.1

# Install essential utilities
echo "Installing essential utilities..."
pip install psutil==7.1.3 pyyaml==6.0.3 numpy==2.3.3

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "
import torch
import sys

print(f'✓ Python version: {sys.version}')
print(f'✓ Torch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import vllm
    print(f'✓ vLLM installed successfully (version: {vllm.__version__})')
except ImportError:
    print('✗ vLLM import failed')
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    print('✓ vLLM components (LLM, SamplingParams) imported successfully')
except ImportError as e:
    print(f'✗ vLLM components import failed: {e}')
    sys.exit(1)

# Check that vLLM automatically installed HuggingFace dependencies
try:
    import transformers
    import tokenizers
    print(f'✓ Transformers version: {transformers.__version__} (via vLLM)')
    print(f'✓ Tokenizers version: {tokenizers.__version__} (via vLLM)')
except ImportError as e:
    print(f'⚠ HuggingFace dependencies not fully installed: {e}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Environment setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "To activate this environment, run:"
    echo "  conda activate ${ENV_NAME}"
    echo ""
    echo "To test the environment, run:"
    echo "  python clarification.py --model llama3.1-8b --topic scifact"
    echo ""
    echo "Note: vLLM provides 3-5x faster inference than HuggingFace pipelines"
else
    echo ""
    echo "=========================================="
    echo "✗ Environment setup failed during verification"
    echo "=========================================="
    exit 1
fi

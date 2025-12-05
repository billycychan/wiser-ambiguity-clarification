"""
vLLM configuration for nvidia/Llama-3.3-70B-Instruct-FP8

Uses vLLM for optimized inference with tensor parallelism across multiple GPUs.
"""

from vllm import LLM
import os
import torch


def create_llm():
    """
    Create and return a vLLM LLM instance for Llama-3.3-70B-Instruct.

    vLLM Configuration:
    - tensor_parallel_size: Distribute across multiple GPUs (can be set via TENSOR_PARALLEL_SIZE env var)
    - gpu_memory_utilization=0.8: Use 80% of GPU memory (conservative for 70B)
    - dtype=auto: Auto-detect dtype from model

    Environment Variables:
    - CUDA_VISIBLE_DEVICES: Specify which GPUs to use (e.g., "0,1" or "2,3")
    - TENSOR_PARALLEL_SIZE: Override tensor parallel size (default: 2)

    Adjust tensor_parallel_size based on your GPU count:
    - 2 GPUs: tensor_parallel_size=2
    - 4 GPUs: tensor_parallel_size=4
    - 8 GPUs: tensor_parallel_size=8

    Returns:
        LLM: vLLM LLM instance for direct generation
    """
    llm = LLM(
        model="nvidia/Llama-3.3-70B-Instruct-FP8",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.7,  # Increase to 90% to get more KV cache memory
        max_model_len=8192,  # Reduce from default 131072 to fit in available memory
        dtype="auto",
        trust_remote_code=True,
    )

    return llm


def get_model_name():
    """Return a short identifier for this model."""
    return "llama3.3-70b-fp8"

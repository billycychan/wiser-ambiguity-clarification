"""
vLLM configuration for meta-llama/Llama-3.1-8B-Instruct

Uses vLLM for optimized inference with PagedAttention and continuous batching.
"""
from vllm import LLM


def create_llm():
    """
    Create and return a vLLM LLM instance for Llama-3.1-8B-Instruct.
    
    vLLM Configuration:
    - tensor_parallel_size=1: Single GPU (8B model fits on one GPU)
    - gpu_memory_utilization=0.9: Use 90% of GPU memory
    - dtype=auto: Let vLLM choose optimal dtype (likely bfloat16 or float16)
    
    Returns:
        LLM: vLLM LLM instance for direct generation
    """
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        dtype="auto",
    )
    
    return llm


def get_model_name():
    """Return a short identifier for this model."""
    return "llama3.1-8b"

"""
vLLM configuration for nvidia/Llama-3.3-70B-Instruct-FP8

Uses vLLM for optimized inference with tensor parallelism across multiple GPUs.
"""

from vllm import LLM


def create_llm():
    """
    Create and return a vLLM LLM instance for Llama-3.3-70B-Instruct.

    vLLM Configuration:
    - tensor_parallel_size=4: Distribute across 4 GPUs (adjust based on your hardware)
    - gpu_memory_utilization=0.85: Use 85% of GPU memory (conservative for 70B)
    - dtype=bfloat16: Use bfloat16 for efficiency (good for H100)

    Note: If your checkpoint has AWQ/GPTQ quantization, vLLM will auto-detect it.
    Otherwise, bfloat16 inference with vLLM's PagedAttention is very efficient.

    Adjust tensor_parallel_size based on your GPU count:
    - 2 GPUs: tensor_parallel_size=2
    - 4 GPUs: tensor_parallel_size=4
    - 8 GPUs: tensor_parallel_size=8

    Returns:
        LLM: vLLM LLM instance for direct generation
    """
    llm = LLM(
        model="nvidia/Llama-3.3-70B-Instruct-FP8",
        tensor_parallel_size=4,  # Adjust based on GPU count
        gpu_memory_utilization=0.5,
        dtype="auto",
        trust_remote_code=True,
    )

    return llm


def format_prompt(query, system_prompt, user_prompt, tokenizer):
    """
    Format the prompt using Llama 3's chat template.

    Args:
        query: The user query
        system_prompt: System instruction
        user_prompt: User prompt template
        tokenizer: Tokenizer instance from vLLM

    Returns:
        str: Formatted prompt ready for generation
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(query=query)},
    ]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted


def get_model_name():
    """Return a short identifier for this model."""
    return "llama3.3-70b-fp8"

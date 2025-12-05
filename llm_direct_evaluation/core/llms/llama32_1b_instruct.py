"""
vLLM configuration for meta-llama/Llama-3.2-1B-Instruct

Uses vLLM for optimized inference.
"""

from vllm import LLM, SamplingParams


def create_llm():
    """
    Create and return a vLLM LLM instance for Llama-3.2-1B-Instruct.

    vLLM Configuration:
    - Single GPU is sufficient for 1B model
    - gpu_memory_utilization=0.5: Use 50% of GPU memory
    - dtype=auto: Let vLLM choose the best dtype

    Returns:
        LLM: vLLM LLM instance for direct generation
    """
    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        dtype="auto",
        trust_remote_code=True,
    )

    return llm


def get_model_name():
    """Return a short identifier for this model."""
    return "llama3.2-1b"


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

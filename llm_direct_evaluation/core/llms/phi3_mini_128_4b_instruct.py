"""
vLLM configuration for microsoft/Phi-3-mini-128k-instruct

Uses vLLM for optimized inference.
"""

from vllm import LLM, SamplingParams


def create_llm():
    """
    Create and return a vLLM LLM instance for Phi-3-mini-128k-instruct.

    vLLM Configuration:
    - Single GPU is sufficient for this model (~3.8B parameters)
    - gpu_memory_utilization=0.9: Use 90% of GPU memory
    - dtype=auto: Let vLLM choose the best dtype

    Returns:
        LLM: vLLM LLM instance for direct generation
    """
    llm = LLM(
        model="microsoft/Phi-3-mini-128k-instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        dtype="auto",
        trust_remote_code=True,
    )

    return llm


def get_model_name():
    """Return a short identifier for this model."""
    return "phi3-mini"


def format_prompt(query, system_prompt, user_prompt, tokenizer):
    """
    Format the prompt using Phi-3's chat template.

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

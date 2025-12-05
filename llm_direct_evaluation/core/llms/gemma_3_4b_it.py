"""
vLLM configuration for google/gemma-2-9b-it (Gemma-3-4B)

Uses vLLM for optimized inference.
"""

import os
from vllm import LLM, SamplingParams


def create_llm():
    """
    Create and return a vLLM LLM instance for Gemma-3-4B.

    vLLM Configuration:
    - Single GPU is sufficient for 4B model
    - gpu_memory_utilization=0.5: Use 50% of GPU memory
    - dtype=auto: Let vLLM choose the best dtype

    Returns:
        LLM: vLLM LLM instance for direct generation
    """
    llm = LLM(
        model="google/gemma-3-4b-it",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        dtype="auto",
        trust_remote_code=True,
    )

    return llm


def get_model_name():
    """Return a short identifier for this model."""
    return "gemma-3-4b"


def format_prompt(query, system_prompt, user_prompt, tokenizer):
    """
    Format the prompt using Gemma's chat template.

    Args:
        query: The user query
        system_prompt: System instruction
        user_prompt: User prompt template
        tokenizer: Tokenizer instance from vLLM

    Returns:
        str: Formatted prompt ready for generation
    """
    # Gemma uses a specific chat format
    messages = [
        {
            "role": "user",
            "content": f"{system_prompt}\n\n{user_prompt.format(query=query)}",
        },
    ]

    # Get tokenizer from the model to apply chat template

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted

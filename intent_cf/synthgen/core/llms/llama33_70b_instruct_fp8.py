"""
Synthetic query generation using Llama-3.1-8B with vLLM.

Migrated from transformers to vLLM for improved performance.
"""
from vllm import LLM, SamplingParams
import os
from tqdm import tqdm
from datetime import datetime

from prompt import get_user_prompt, SYSTEM_PROMPT
from topics import topics


def build_prompt(tokenizer, topic: str) -> str:
    """Build prompt for synthetic query generation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(topic)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_llm_and_tokenizer(model: str):
    """Initialize vLLM and tokenizer."""
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        dtype="auto",
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    return llm, tok


def main():
    model = "meta-llama/Llama-3.1-8B-Instruct"
    llm, tok = get_llm_and_tokenizer(model)

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/llama31_8b_instruct_{timestamp}.log"

    print(f"Logging responses to: {log_file}")

    # vLLM sampling parameters
    sampling_params = SamplingParams(
        max_tokens=5000,
        temperature=1.0,
        top_p=0.95,
    )

    with open(log_file, "w", encoding="utf-8") as log:
        for topic in tqdm(topics, desc="Generating queries"):
            # Build prompt
            prompt = build_prompt(tok, topic)

            # Generate response using vLLM
            outputs = llm.generate([prompt], sampling_params)
            
            # Extract generated text
            response = outputs[0].outputs[0].text

            log.write(f"{response}\n")
            log.flush()  # Ensure it's written immediately

    print(f"Completed! Log saved to: {log_file}")


if __name__ == "__main__":
    main()

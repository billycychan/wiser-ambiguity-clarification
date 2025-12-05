"""
Synthetic query generation using Llama-3.3-70B with vLLM.

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
        tensor_parallel_size=4,  # Use 4 GPUs for 70B model
        gpu_memory_utilization=0.5,
        dtype="auto",
        trust_remote_code=True,
        enforce_eager=True,  # Disable CUDA graphs to avoid initialization issues
    )
    tok = llm.get_tokenizer()
    return llm, tok


def main():
    model = "nvidia/Llama-3.3-70B-Instruct-FP8"
    llm, tok = get_llm_and_tokenizer(model)

    # Create output directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("../data", exist_ok=True)

    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output files
    tsv_file = f"../data/Llama-3.3-70B-Instruct_balanced_strict_{timestamp}.tsv"
    log_file = f"logs/llama33_70b_instruct_{timestamp}.log"

    print(f"Generating synthetic queries...")
    print(f"TSV output: {tsv_file}")
    print(f"Raw log: {log_file}")

    # vLLM sampling parameters
    sampling_params = SamplingParams(
        max_tokens=5000,
        temperature=1.0,
        top_p=0.95,
    )

    # Open both files for writing
    with open(tsv_file, "w", encoding="utf-8") as tsv, open(
        log_file, "w", encoding="utf-8"
    ) as log:

        # Write TSV header
        tsv.write("topic\tinitial_request\tbinary_label\tuser_information_need\n")

        for topic in tqdm(topics, desc="Generating queries"):
            # Build prompt
            prompt = build_prompt(tok, topic)

            # Generate response using vLLM
            outputs = llm.generate([prompt], sampling_params)

            # Extract generated text
            response = outputs[0].outputs[0].text

            # Log raw response
            log.write(f"=== {topic} ===\n")
            log.write(f"{response}\n\n")
            log.flush()

            # Parse and write to TSV
            # The response should contain tab-separated lines
            for line in response.strip().split("\n"):
                line = line.strip()
                if line:  # Skip empty lines
                    # Write to TSV file
                    tsv.write(f"{line}\n")
                    tsv.flush()

    print(f"\nCompleted!")
    print(f"TSV file saved to: {tsv_file}")
    print(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    main()

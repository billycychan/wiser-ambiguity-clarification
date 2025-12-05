"""
Synthetic query generation using vLLM with selectable models and batch sizes.
"""

import argparse
import re
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
from vllm import SamplingParams

# Add project root to path to access llm_direct_evaluation
project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, project_root)

# Import model modules from llm_direct_evaluation
from llm_direct_evaluation.core.llms import phi3_mini_128_4b_instruct as phi3
from llm_direct_evaluation.core.llms import llama31_8b_instruct as llama31_8b
from llm_direct_evaluation.core.llms import llama33_70b_instruct_fp8 as llama33_70b
from llm_direct_evaluation.core.llms import gemma_3_27b_it as gemma27b

# Import synthgen components
from prompt import get_user_prompt, SYSTEM_PROMPT
from topics import topics

# Model registry - maps model names to their modules
model_modules = {
    "Phi-3": phi3,
    "Llama-3.1-8B": llama31_8b,
    "Llama-3.3-70B": llama33_70b,
    "Gemma-3-27B": gemma27b,
}


def build_prompt(tokenizer, topic: str) -> str:
    """Builtic query generation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(topic)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_response_line(line):
    """Parse a line from the model response."""
    # Try splitting by tab first
    parts = line.strip().split("\t")
    if len(parts) == 4:
        return parts

    # Try regex for flexible whitespace
    # Pattern: Topic (any) <spaces> Query (any) <spaces> Label (0/1) <spaces> Intent (any)
    # We use non-greedy matching for topic and query, but greedy for spaces
    match = re.match(r"^(.*?)\s{2,}(.*?)\s{2,}([01])\s{2,}(.*)$", line.strip())
    if match:
        return match.groups()

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic queries using vLLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(model_modules.keys()),
        required=True,
        help="Model to use for generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Directory to save output files (default: ../data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of topics to process (for testing)",
    )
    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batch_size

    print(f"Initializing {model_name}...")
    model_module = model_modules[model_name]
    llm = model_module.create_llm()
    tokenizer = llm.get_tokenizer()

    # Create output directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output files
    safe_model_name = model_name.replace(" ", "_").replace("-", "_")
    tsv_file = os.path.join(
        args.output_dir, f"{safe_model_name}_balanced_strict_{timestamp}.tsv"
    )
    log_file = f"logs/{safe_model_name}_{timestamp}.log"

    print(f"Generating synthetic queries...")
    print(f"Model: {model_name}")
    print(f"Batch Size: {batch_size}")
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

        processing_topics = topics
        if args.limit:
            processing_topics = topics[: args.limit]

        total_topics = len(processing_topics)

        # Process in batches
        for i in tqdm(range(0, total_topics, batch_size), desc="Generating queries"):
            batch_topics = processing_topics[i : i + batch_size]

            # Build prompts for the batch
            prompts = [build_prompt(tokenizer, topic) for topic in batch_topics]

            # Generate responses using vLLM
            outputs = llm.generate(prompts, sampling_params)

            for j, output in enumerate(outputs):
                topic = batch_topics[j]
                response = output.outputs[0].text

                # Log raw response
                log.write(f"=== {topic} ===\n")
                log.write(f"{response}\n\n")
                log.flush()

                # Parse and write to TSV
                for line in response.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    parsed = parse_response_line(line)
                    if parsed:
                        # Unpack parsed parts; ignore the topic from model, use the requested topic
                        _, query, binary_label, user_information_need = parsed

                        # Write to TSV file with correct topic
                        tsv.write(
                            f"{topic}\t{query}\t{binary_label}\t{user_information_need}\n"
                        )
                        tsv.flush()
                    else:
                        # Log warning for unparseable line
                        log.write(f"WARNING: Could not parse line: {line}\n")

    print(f"\nCompleted!")
    print(f"TSV file saved to: {tsv_file}")
    print(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    main()

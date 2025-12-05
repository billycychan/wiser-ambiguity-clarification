from prompts import (
    SYSTEM_PROMPT_ZERO_SHOT,
    SYSTEM_PROMPT_FEW_SHOT,
    SYSTEM_PROMPT_AMBIG2DOC,
    USER_PROMPT_TEMPLATE,
)
import llm
import csv
import os
import argparse
from datetime import datetime
import time
import logging

# Configuration
DEFAULT_BATCH_SIZE = 32


# Generate output path from input path
def get_output_path(input_path, prompt_type, model_name, timestamp):
    """Generate output path by adding prompt type, model name, and timestamp to the filename."""
    dir_name = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    # Remove extension
    name_without_ext = os.path.splitext(file_name)[0]
    ext = os.path.splitext(file_name)[1]
    # Add prompt type, model name, and timestamp
    output_name = (
        f"clarified_{name_without_ext}_{prompt_type}_{model_name}_{timestamp}{ext}"
    )
    return os.path.join(dir_name, output_name) if dir_name else output_name


def setup_logging(log_path):
    """Setup logging to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def format_prompt(query, system_prompt, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def process_tsv_file(
    input_path, output_path, system_prompt, llm, tokenizer, batch_size=32
):
    """
    Read queries from TSV file, clarify them using the LLM, and save results.

    Args:
        input_path: Path to input TSV file with ID and query columns
        output_path: Path to output TSV file for clarified queries
        system_prompt: System prompt to use for the LLM
        llm: The vLLM LLM instance to use
        tokenizer: The tokenizer for the LLM
        batch_size: Number of queries to process in each batch
    """
    logger = logging.getLogger()

    # Read all queries from input file
    queries_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                query_id = row[0]
                query_text = row[1]
                queries_data.append((query_id, query_text))

    logger.info(f"Loaded {len(queries_data)} queries from {input_path}")

    # Process queries in batches
    clarified_results = []
    total_inference_time = 0
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i : i + batch_size]
        batch_queries = [query_text for _, query_text in batch]
        batch_ids = [query_id for query_id, _ in batch]

        # Format prompts for batch
        formatted_prompts = [
            format_prompt(query, system_prompt, tokenizer) for query in batch_queries
        ]

        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(queries_data)-1)//batch_size + 1} ({len(batch)} queries)..."
        )

        # Run inference with vLLM
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0.0,  # Greedy decoding (do_sample=False)
            skip_special_tokens=True,
        )

        start_time = time.time()
        outputs = llm.generate(
            formatted_prompts,
            sampling_params,
        )
        batch_time = time.time() - start_time
        total_inference_time += batch_time

        logger.info(
            f"  Batch inference time: {batch_time:.2f}s ({batch_time/len(batch):.3f}s per query)"
        )

        # Extract clarified queries from vLLM outputs
        for query_id, query_text, output in zip(batch_ids, batch_queries, outputs):
            # vLLM returns RequestOutput objects with .outputs list
            generated_text = output.outputs[0].text

            # Clean up and store result - replace newlines with spaces for single-line TSV format
            clarified = generated_text.strip().replace("\n", " ").replace("\r", " ")
            # Remove multiple consecutive spaces
            clarified = " ".join(clarified.split())
            clarified_results.append((query_id, clarified))

    # Write results to output file
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for query_id, clarified_query in clarified_results:
            writer.writerow([query_id, clarified_query])

    logger.info(f"\nSaved {len(clarified_results)} clarified queries to {output_path}")
    logger.info(f"Total inference time: {total_inference_time:.2f}s")
    logger.info(
        f"Average time per query: {total_inference_time/len(clarified_results):.3f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clarify queries using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(llm.list_available_models())}",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=llm.list_available_models(),
        default="llama3.1-8b",
        help="Model to use for clarification (default: llama3.1-8b)",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["zero-shot", "few-shot", "ambig2doc"],
        default="few-shot",
        help="Type of prompt to use (default: few-shot)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="scifact",
        help="Topic dataset to use (e.g., scifact, scidocs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    # Load the selected model pipeline
    logger_temp = logging.getLogger()
    logger_temp.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger_temp.addHandler(console_handler)

    logger_temp.info(f"Loading model: {args.model}...")
    llm, model_id = llm.get_llm(args.model)
    tokenizer = llm.get_tokenizer()
    logger_temp.info(f"Model loaded successfully!")

    # Select system prompt based on prompt type
    if args.prompt_type == "few-shot":
        system_prompt = SYSTEM_PROMPT_FEW_SHOT
    elif args.prompt_type == "ambig2doc":
        system_prompt = SYSTEM_PROMPT_AMBIG2DOC
    else:
        system_prompt = SYSTEM_PROMPT_ZERO_SHOT

    # Generate input file path
    input_file = f"queries/topics.beir-v1.0.0-{args.topic}.test.tsv"

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate output file path with model name
    output_file = get_output_path(input_file, args.prompt_type, model_id, timestamp)

    # Generate log file path in /logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.basename(output_file).replace(".tsv", ".log")
    log_file = os.path.join(log_dir, log_filename)

    # Setup logging
    logger = setup_logging(log_file)

    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model} ({model_id})")
    logger.info(f"  Prompt type: {args.prompt_type}")
    logger.info(f"  Topic: {args.topic}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Input file: {input_file}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info("")

    # Process the file
    process_tsv_file(
        input_file,
        output_file,
        system_prompt,
        llm,
        tokenizer,
        batch_size=args.batch_size,
    )

    # Cleanup to prevent "Engine core died unexpectedly" error
    import gc

    del llm
    gc.collect()

"""
LLM Inference Evaluation using vLLM

This script evaluates Large Language Models (LLMs) on ambiguity detection tasks
using vLLM for optimized inference.

"""

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
import sys
import os
import argparse
from datetime import datetime
import time
from tqdm import tqdm
from vllm import SamplingParams
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Import model modules
import core.llms.phi3_mini_128_4b_instruct as phi3
import core.llms.llama31_8b_instruct as llama31_8b
import core.llms.llama32_3b_instruct as llama32_3b
import core.llms.llama32_1b_instruct as llama32_1b
import core.llms.llama33_70b_instruct_fp8 as llama33_70b
import core.llms.gemma_3_1b_it as gemma1b
import core.llms.gemma_3_4b_it as gemma4b
import core.llms.gemma_3_27b_it as gemma27b

# Import prompts
from core.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_FEW_SHOT, USER_PROMPT_TEMPLATE

# Import utilities
from evaluations.llm_inference.utils import (
    normalize_output,
    seed_everything,
    setup_logging,
    save_results,
)

logger = logging.getLogger(__name__)


# Model registry - maps model names to their modules
model_modules = {
    "Phi-3": phi3,
    "Llama-3.1-8B": llama31_8b,
    "Llama-3.2-1B": llama32_1b,
    "Llama-3.2-3B": llama32_3b,
    "Llama-3.3-70B": llama33_70b,
    "Gemma-3-1B": gemma1b,
    "Gemma-3-4B": gemma4b,
    "Gemma-3-27B": gemma27b,
}

# Dataset paths
datasets = {
    "ambignq": os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "ambignq_preprocessed.tsv"
    ),
    "clariq": os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "clariq_preprocessed.tsv"
    ),
}


def run_inference_vllm(
    model_name,
    llm,
    questions,
    system_prompt,
    user_prompt,
    max_new_tokens,
    temperature,
    do_sample,
    batch_size,
):
    """
    Run inference using vLLM.

    Args:
        model_name: Name of the model
        llm: vLLM LLM instance
        questions: List of questions
        system_prompt: System prompt
        user_prompt: User prompt template
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        batch_size: Batch size for inference

    Returns:
        tuple: (predictions, inference_times)
    """
    model_module = model_modules[model_name]
    formatter = model_module.format_prompt

    predictions = []
    inference_times = []
    total_questions = len(questions)

    logger.info(f"Starting inference for {model_name} on {total_questions} questions")
    logger.debug(
        f"Inference parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}, batch_size={batch_size}"
    )

    # Create vLLM sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature if do_sample else 0.0,
        max_tokens=max_new_tokens,
        top_p=0.95 if do_sample else 1.0,
    )

    with tqdm(total=total_questions, desc=f"Inference {model_name}") as pbar:
        for i in range(0, total_questions, batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_questions + batch_size - 1) // batch_size

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)"
            )

            # Get tokenizer from vLLM instance
            tokenizer = llm.get_tokenizer()

            # Format prompts for batch
            prompts_batch = [
                formatter(q, system_prompt, user_prompt, tokenizer)
                for q in batch_questions
            ]

            # Run vLLM inference
            start_time = time.time()
            outputs = llm.generate(prompts_batch, sampling_params)
            end_time = time.time()

            batch_time = end_time - start_time
            time_per_query = batch_time / len(batch_questions)

            logger.debug(
                f"Batch {batch_num} inference time: {batch_time:.4f}s ({time_per_query:.4f}s per query)"
            )

            # Process outputs
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                pred = normalize_output(generated_text)
                predictions.append(pred)
                inference_times.append(time_per_query)

            pbar.update(len(batch_questions))

    logger.info(
        f"Completed inference for {model_name}. Processed {len(predictions)} predictions"
    )
    return predictions, inference_times


def evaluate_model(model_name, dataset_name, df, system_prompt, user_prompt, args):
    """
    Evaluate a model on a dataset.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        df: DataFrame with evaluation data
        system_prompt: System prompt
        user_prompt: User prompt template
        args: Command-line arguments

    Returns:
        list: Predictions
    """
    logger.info(f"=== Starting evaluation of {model_name} on {dataset_name} ===")
    print(f"\nEvaluating {model_name} on {dataset_name}")

    # Get model module and create LLM instance
    model_module = model_modules[model_name]
    logger.info(f"Loading {model_name} with vLLM...")
    llm = model_module.create_llm()
    logger.info(f"{model_name} loaded successfully")

    questions = df["initial_request"].tolist()
    true_labels = df["binary_label"].tolist()
    logger.debug(f"Dataset size: {len(questions)} samples")

    # Run inference
    predictions, inference_times = run_inference_vllm(
        model_name,
        llm,
        questions,
        system_prompt,
        user_prompt,
        args.max_new_tokens,
        args.temperature,
        args.do_sample,
        args.batch_size,
    )

    # Compute metrics
    report = classification_report(
        true_labels, predictions, target_names=["Not Ambiguous", "Ambiguous"], digits=4
    )
    
    # Calculate weighted metrics explicitly (accounts for class imbalance)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    accuracy = (
        sum([int(p == t) for p, t in zip(predictions, true_labels)]) / len(true_labels)
        if true_labels
        else 0.0
    )

    print(f"\n{model_name} on {dataset_name}:")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\n--- Weighted Metrics (for imbalanced dataset) ---")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1-Score:  {weighted_f1:.4f}")

    logger.info(f"Evaluation results - Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted Metrics - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")

    # Save results
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    predictions_dir = os.path.join(os.path.dirname(__file__), "..", "..", "predictions")
    save_results(
        model_name,
        dataset_name,
        df,
        predictions,
        inference_times,
        report,
        accuracy,
        weighted_precision,
        weighted_recall,
        weighted_f1,
        args,
        logs_dir,
        predictions_dir,
    )

    logger.info(f"=== Completed evaluation of {model_name} on {dataset_name} ===\n")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on ambiguity detection using vLLM."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(model_modules.keys()),
        default=list(model_modules.keys()),
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(datasets.keys()),
        default=list(datasets.keys()),
        help="Datasets to use (default: all)",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="zero_shot",
        help="Type of system prompt (zero_shot, few_shot, or custom; default: zero_shot)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5,
        help="Max new tokens for generation (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)",
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Enable sampling (default: False)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Determine system prompt
    prompt_map = {
        "few_shot": SYSTEM_PROMPT_FEW_SHOT,
        "zero_shot": SYSTEM_PROMPT,
    }
    system_prompt = prompt_map.get(args.prompt_type, "You are a helpful assistant.")

    seed_everything(args.seed)

    results = {}
    for model_name in args.models:
        for dataset_name in args.datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            df = pd.read_csv(datasets[dataset_name], sep="\t")
            preds = evaluate_model(
                model_name, dataset_name, df, system_prompt, USER_PROMPT_TEMPLATE, args
            )
            results[(model_name, dataset_name)] = preds

    logger.info("=" * 80)
    logger.info("Evaluation run completed successfully")
    logger.info("=" * 80)

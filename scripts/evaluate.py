import pandas as pd
from sklearn.metrics import classification_report
import re
import sys
import os
import argparse
from datetime import datetime
import time
from tqdm import tqdm

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import models
import core.llms.phi3_mini_128_8b_instruct as phi3
import core.llms.llama31_8b_instruct as llama31_8b
import core.llms.llama32_3b_instruct as llama32_3b
import core.llms.llama32_1b_instruct as llama32_1b
import core.llms.gemma_3_1b_it as gemma1b
import core.llms.gemma_3_4b_it as gemma4b
import core.llms.gemma_3_27b_it as gemma27b

# Import prompts
from core.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_FEW_SHOT, USER_PROMPT_TEMPLATE
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    logger.info(f"Setting global random seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


models = {
    "Phi-3": phi3.pipeline,
    "Llama-3.1-8B": llama31_8b.pipeline,
    "Llama-3.2-1B": llama32_1b.pipeline,
    "Llama-3.2-3B": llama32_3b.pipeline,
    "Gemma-3-1B": gemma1b.pipeline,
    "Gemma-3-4B": gemma4b.pipeline,
    "Gemma-3-27B": gemma27b.pipeline,
}

model_formatters = {
    "Phi-3": phi3.format_prompt,
    "Llama-3.1-8B": llama31_8b.format_prompt,
    "Llama-3.2-1B": llama32_1b.format_prompt,
    "Llama-3.2-3B": llama32_3b.format_prompt,
    "Gemma-3-1B": gemma1b.format_prompt,
    "Gemma-3-4B": gemma4b.format_prompt,
    "Gemma-3-27B": gemma27b.format_prompt,
}

datasets = {
    # 'ambignq': os.path.join(os.path.dirname(__file__), '..', 'data', 'ambignq_preprocessed.tsv'),
    "clariq": os.path.join(
        os.path.dirname(__file__), "..", "data", "clariq_preprocessed.tsv"
    )
}


def normalize_output(output_text):
    response = output_text.lower().strip()

    # Find all occurrences of 'yes' or 'no' (case insensitive, word boundaries)
    matches = re.findall(r"\b(yes|no)\b", response, re.IGNORECASE)

    if matches:
        # Return 1 if last occurrence is 'yes', 0 if 'no'
        return 1 if matches[-1].lower() == "yes" else 0
    else:
        # Otherwise return 0
        return 0


def run_inference(
    model_name,
    questions,
    system_prompt,
    user_prompt,
    max_new_tokens,
    temperature,
    do_sample,
    batch_size,
):
    pipeline = models[model_name]
    if pipeline.tokenizer.pad_token is None:
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    formatter = model_formatters[model_name]
    predictions = []
    inference_times = []
    total_questions = len(questions)
    with tqdm(total=total_questions, desc=f"Inference {model_name}") as pbar:
        for i in range(0, total_questions, batch_size):
            batch_questions = questions[i : i + batch_size]
            prompts_batch = [
                formatter(q, system_prompt, user_prompt) for q in batch_questions
            ]
            start_time = time.time()
            results = pipeline(
                prompts_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            end_time = time.time()
            time_per_query = (end_time - start_time) / len(batch_questions)
            for _ in batch_questions:
                inference_times.append(time_per_query)
            for j, result_list in enumerate(results):
                generated = (
                    result_list[0]["generated_text"]
                    .replace(prompts_batch[j], "")
                    .strip()
                )
                pred = normalize_output(generated)
                predictions.append(pred)
            pbar.update(len(batch_questions))
    return predictions, inference_times


def evaluate_model(model_name, dataset_name, df, system_prompt, user_prompt, args):
    print(f"\nEvaluating {model_name} on {dataset_name}")
    questions = df["initial_request"].tolist()
    true_labels = df["binary_label"].tolist()
    predictions, inference_times = run_inference(
        model_name,
        questions,
        system_prompt,
        user_prompt,
        args.max_new_tokens,
        args.temperature,
        args.do_sample,
        args.batch_size,
    )
    report = classification_report(
        true_labels, predictions, target_names=["Not Ambiguous", "Ambiguous"]
    )
    accuracy = (
        sum([int(p == t) for p, t in zip(predictions, true_labels)]) / len(true_labels)
        if true_labels
        else 0.0
    )
    print(f"\n{model_name} on {dataset_name}:")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")

    # Compute inference time stats
    if inference_times:
        min_time = min(inference_times)
        max_time = max(inference_times)
        mean_time = sum(inference_times) / len(inference_times)
    else:
        min_time = max_time = mean_time = 0.0

    # Create logs folder
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Sanitize names for filename
    model_safe = re.sub(r"[^\w]", "_", model_name).lower()
    dataset_safe = dataset_name.lower()
    prompt_type_safe = args.prompt_type.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save report to log file
    log_filename = f"{model_safe}_{dataset_safe}_{prompt_type_safe}_{timestamp}.log"
    with open(os.path.join(logs_dir, log_filename), "w") as f:
        f.write(
            f"Model: {model_name}\n"
            f"Dataset: {dataset_name}\n"
            f"Timestamp: {timestamp}\n"
            f"Prompt Type: {args.prompt_type}\n"
            f"Parameters: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, do_sample={args.do_sample}, batch_size={args.batch_size}, seed={args.seed}\n\n"
            f"Classification Report:\n{report}\n\n"
            f"Accuracy: {accuracy:.4f}\n\n"
            f"Inference Time Stats:\nMin: {min_time:.4f}s\nMax: {max_time:.4f}s\nMean: {mean_time:.4f}s\n"
        )

    # Save predictions to TSV
    pred_filename = (
        f"{model_safe}_{dataset_safe}_{prompt_type_safe}_{timestamp}_predictions.tsv"
    )
    df["predicted_label"] = predictions
    df.to_csv(os.path.join(logs_dir, pred_filename), sep="\t", index=False)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on ambiguity detection."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=models.keys(),
        default=list(models.keys()),
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=datasets.keys(),
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
        help="Max new tokens for generation (default: 50)",
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
        default=4247918,
        help="Random seed for reproducibility (default: 424247918)",
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
            df = pd.read_csv(datasets[dataset_name], sep="\t")
            preds = evaluate_model(
                model_name, dataset_name, df, system_prompt, USER_PROMPT_TEMPLATE, args
            )
            results[(model_name, dataset_name)] = preds

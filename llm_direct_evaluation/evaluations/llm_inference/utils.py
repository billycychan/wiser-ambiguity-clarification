"""
Shared utilities for LLM inference evaluation.
"""

import re
import os
import random
import numpy as np
import torch
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_dir, log_file):
    """
    Setup logging to console and file.

    Args:
        log_dir: Directory for log files
        log_file: Log filename
    """
    os.makedirs(log_dir, exist_ok=True)

    # Clear existing handlers
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def seed_everything(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    logger.info(f"Setting global random seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def normalize_output(output_text):
    """
    Normalize LLM output to binary label (0 or 1).

    Looks for 'yes' or 'no' in the response and returns 1 for ambiguous (yes)
    or 0 for not ambiguous (no).

    Args:
        output_text: Generated text from LLM

    Returns:
        int: 1 if ambiguous (yes), 0 if not ambiguous (no)
    """
    response = output_text.lower().strip()

    # Find all occurrences of 'yes' or 'no' (case insensitive, word boundaries)
    matches = re.findall(r"\b(yes|no)\b", response, re.IGNORECASE)

    if matches:
        # Return 1 if last occurrence is 'yes', 0 if 'no'
        return 1 if matches[-1].lower() == "yes" else 0
    else:
        # Otherwise return 0
        return 0


def save_results(
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
):
    """
    Save evaluation results to log files and TSV.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        df: DataFrame with evaluation data
        predictions: List of predictions
        inference_times: List of inference times per sample
        report: Classification report string
        accuracy: Accuracy score
        weighted_precision: Weighted precision score
        weighted_recall: Weighted recall score
        weighted_f1: Weighted F1 score
        args: Argparse arguments
        logs_dir: Directory for saving log files
        predictions_dir: Directory for saving prediction files
    """
    # Compute inference time stats
    if inference_times:
        min_time = min(inference_times)
        max_time = max(inference_times)
        mean_time = sum(inference_times) / len(inference_times)
        logger.info(
            f"Inference time stats - Min: {min_time:.4f}s, Max: {max_time:.4f}s, Mean: {mean_time:.4f}s"
        )
    else:
        min_time = max_time = mean_time = 0.0
        logger.warning("No inference times recorded")

    # Create logs and predictions folders
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Sanitize names for filename
    model_safe = re.sub(r"[^\w]", "_", model_name).lower()
    dataset_safe = dataset_name.lower()
    prompt_type_safe = args.prompt_type.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save report to log file
    log_filename = f"{model_safe}_{dataset_safe}_{prompt_type_safe}_{timestamp}.log"
    logger.info(f"Saving evaluation report to {log_filename}")
    with open(os.path.join(logs_dir, log_filename), "w") as f:
        f.write(
            f"Model: {model_name}\n"
            f"Dataset: {dataset_name}\n"
            f"Timestamp: {timestamp}\n"
            f"Prompt Type: {args.prompt_type}\n"
            f"Parameters: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, do_sample={args.do_sample}, batch_size={args.batch_size}, seed={args.seed}\n\n"
            f"Classification Report:\n{report}\n\n"
            f"Accuracy: {accuracy:.4f}\n\n"
            f"--- Weighted Metrics (for imbalanced dataset) ---\n"
            f"Weighted Precision: {weighted_precision:.4f}\n"
            f"Weighted Recall:    {weighted_recall:.4f}\n"
            f"Weighted F1-Score:  {weighted_f1:.4f}\n\n"
            f"Inference Time Stats:\nMin: {min_time:.4f}s\nMax: {max_time:.4f}s\nMean: {mean_time:.4f}s\n"
        )

    # Save predictions to TSV
    pred_filename = (
        f"{model_safe}_{dataset_safe}_{prompt_type_safe}_{timestamp}_predictions.tsv"
    )
    logger.info(f"Saving predictions to {pred_filename}")
    df["predicted_label"] = predictions
    df.to_csv(os.path.join(predictions_dir, pred_filename), sep="\t", index=False)

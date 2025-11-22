# ...existing code...
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_PREFIX = "roberta-baseline-zeroshot"  # Zero-shot baseline name for output

# Dataset paths
CLARIQ_DATA_PATH = "/nfs/u40/chanc187/source/eval_cnp/data/clariq_preprocessed.tsv"
AMBIGNQ_DATA_PATH = "/nfs/u40/chanc187/source/eval_cnp/data/ambignq_preprocessed.tsv"

# Output directory
OUTPUT_DIR = "./logs/roberta"

# Prototypes used for zero-shot similarity (index 0 => not ambiguous, index 1 => ambiguous)
PROTOTYPES = [
    "This is a clear, specific, unambiguous request.",  # class 0: not ambiguous
    "This request is vague, incomplete, or ambiguous.",  # class 1: ambiguous
]
# ============================================================================


def load_encoder(model_name="roberta-base", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_texts(texts, tokenizer, model, device, batch_size=32):
    """
    Encode texts into CLS-token embeddings using RoBERTa.
    Returns np.ndarray with shape (N, hidden_size). Handles empty lists.
    """
    if len(texts) == 0:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use <s>/CLS token embedding (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_emb.cpu().numpy())
    return np.vstack(embeddings)


def predict_zero_shot(df, tokenizer, model, device, batch_size=32):
    """
    Zero-shot prediction using cosine similarity to label prototypes.
    Input dataframe must have 'initial_request' and 'binary_label' columns.
    """
    texts = df["initial_request"].astype(str).tolist()
    labels = df["binary_label"].astype(int).tolist()

    X = encode_texts(texts, tokenizer, model, device, batch_size)
    prototype_embeddings = encode_texts(
        PROTOTYPES, tokenizer, model, device, batch_size=2
    )

    if X.shape[0] == 0:
        return np.array([], dtype=int), labels

    sims = cosine_similarity(X, prototype_embeddings)  # shape (N, 2)
    preds = sims.argmax(axis=1)
    return preds, labels


def compute_metrics(labels, preds, target_names=None):
    """
    Compute and return a metrics dict and classification report string.
    """
    if target_names is None:
        target_names = ["not ambiguous", "ambiguous"]

    report = classification_report(labels, preds, target_names=target_names)
    metrics = {
        "classification_report": report,
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }
    return metrics


def save_report_and_predictions(
    out_dir, prefix, dataset_name, df, labels, preds, metrics, timestamp
):
    report_path = os.path.join(out_dir, f"{prefix}_report_{timestamp}.txt")
    with open(report_path, "a") as f:
        f.write(f"Classification Report for {dataset_name}:\n")
        f.write(metrics["classification_report"])
        f.write(f"\nAccuracy: {metrics['accuracy']}\n")
        f.write(f"F1: {metrics['f1']}\n")
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Recall: {metrics['recall']}\n\n")

    out_df = pd.DataFrame(
        {
            "initial_request": df["initial_request"],
            "binary_label": labels,
            "prediction": preds,
        }
    )
    tsv_path = os.path.join(
        out_dir, f"{prefix}_{dataset_name.lower()}_{timestamp}_predictions.tsv"
    )
    out_df.to_csv(tsv_path, sep="\t", index=False)
    return report_path, tsv_path


def evaluate_dataset(df, dataset_name, tokenizer, model, device):
    """
    Run zero-shot prediction and compute runtime + metrics.
    Returns preds, labels, metrics, elapsed_time.
    """
    start_time = time.time()
    preds, labels = predict_zero_shot(df, tokenizer, model, device)
    elapsed = time.time() - start_time
    metrics = compute_metrics(labels, preds)
    return preds, labels, metrics, elapsed


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    clariq_df = pd.read_csv(CLARIQ_DATA_PATH, sep="\t")
    ambignq_df = pd.read_csv(AMBIGNQ_DATA_PATH, sep="\t")

    # Encoder
    print("Loading RoBERTa model...")
    tokenizer, model, device = load_encoder()
    print(f"Model loaded on device: {device}")

    # ClariQ
    print("\nEvaluating on ClariQ (Zero-Shot)...")
    preds_clariq, labels_clariq, metrics_clariq, time_clariq = evaluate_dataset(
        clariq_df, "ClariQ", tokenizer, model, device
    )
    num_samples = len(clariq_df) if len(clariq_df) > 0 else 1
    avg_time = time_clariq / num_samples

    print("Classification Report for ClariQ:")
    print(metrics_clariq["classification_report"])
    print(f"Accuracy: {metrics_clariq['accuracy']}")
    print(f"F1: {metrics_clariq['f1']}")
    print(f"Precision: {metrics_clariq['precision']}")
    print(f"Recall: {metrics_clariq['recall']}")
    print(f"Inference Time: {time_clariq} seconds")
    print(f"Average Inference Time per Sample: {avg_time} seconds")

    # AmbigNQ
    print("\nEvaluating on AmbigNQ (Zero-Shot)...")
    preds_ambignq, labels_ambignq, metrics_ambignq, time_ambignq = evaluate_dataset(
        ambignq_df, "AmbigNQ", tokenizer, model, device
    )
    num_samples_ambignq = len(ambignq_df) if len(ambignq_df) > 0 else 1
    avg_time_ambignq = time_ambignq / num_samples_ambignq

    print("Classification Report for AmbigNQ:")
    print(metrics_ambignq["classification_report"])
    print(f"Accuracy: {metrics_ambignq['accuracy']}")
    print(f"F1: {metrics_ambignq['f1']}")
    print(f"Precision: {metrics_ambignq['precision']}")
    print(f"Recall: {metrics_ambignq['recall']}")
    print(f"Inference Time: {time_ambignq} seconds")
    print(f"Average Inference Time per Sample: {avg_time_ambignq} seconds")

    # Save aggregated report file (create header so subsequent saves append cleanly)
    report_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_report_{timestamp}.txt")
    with open(report_path, "w") as fh:
        fh.write(f"Zero-shot RoBERTa baseline report created at {timestamp}\n\n")

    _, clariq_tsv = save_report_and_predictions(
        OUTPUT_DIR,
        OUTPUT_PREFIX,
        "ClariQ",
        clariq_df,
        labels_clariq,
        preds_clariq,
        metrics_clariq,
        timestamp,
    )
    _, ambignq_tsv = save_report_and_predictions(
        OUTPUT_DIR,
        OUTPUT_PREFIX,
        "AmbigNQ",
        ambignq_df,
        labels_ambignq,
        preds_ambignq,
        metrics_ambignq,
        timestamp,
    )

    print(f"Saved ClariQ predictions to {clariq_tsv}")
    print(f"Saved AmbigNQ predictions to {ambignq_tsv}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

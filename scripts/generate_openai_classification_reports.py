import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import os
import re
from datetime import datetime


def generate_report(tsv_path):
    # Load the TSV
    df = pd.read_csv(tsv_path, sep="\t")
    true_labels = df["binary_label"].tolist()
    predicted_labels = df["predicted_label"].tolist()

    # Compute classification report
    report = classification_report(
        true_labels, predicted_labels, target_names=["Not Ambiguous", "Ambiguous"]
    )
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Extract info from filename
    filename = os.path.basename(tsv_path)
    # Example: gpt-4.1_ambignq_few_shot_20251118_011607_predictions.tsv
    match = re.match(r"(.+)_(.+)_(few_shot|zero_shot)_(\d+)_predictions\.tsv", filename)
    if match:
        model_name = match.group(1).replace("_", "-").replace("gpt-4.1", "GPT-4.1")
        dataset = match.group(2)
        prompt_type = match.group(3).replace("_", " ")
        timestamp = match.group(4)
    else:
        # Fallback
        model_name = filename.split("_")[0]
        dataset = "unknown"
        prompt_type = "unknown"
        timestamp = "unknown"

    # Create log content
    log_content = f"""Model: {model_name}
Dataset: {dataset}
Timestamp: {timestamp}
Prompt Type: {prompt_type}
Parameters: max_new_tokens=5, temperature=0.0, do_sample=False, batch_size=64, seed=0

Classification Report:
{report}

Accuracy: {accuracy:.4f}

Inference Time Stats:
Min: N/A
Max: N/A
Mean: N/A
"""

    # Save log file
    log_filename = filename.replace("_predictions.tsv", ".log")
    log_path = os.path.join(os.path.dirname(tsv_path), log_filename)
    with open(log_path, "w") as f:
        f.write(log_content)
    print(f"Generated report: {log_path}")


if __name__ == "__main__":
    results_dir = "results"
    for file in os.listdir(results_dir):
        if file.endswith("_predictions.tsv") and "gpt" in file:
            tsv_path = os.path.join(results_dir, file)
            generate_report(tsv_path)

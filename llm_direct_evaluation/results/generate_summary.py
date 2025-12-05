"""
Generate summary table from evaluation logs.

This script parses .log files in the logs/ directory and generates a markdown table
summarizing the performance metrics (Precision, Recall, F1, Latency) for each model.
"""

import os
import re
import pandas as pd
import glob
import argparse
from datetime import datetime


def parse_log_file(file_path):
    """Parse a single log file to extract metrics."""
    with open(file_path, "r") as f:
        content = f.read()

    # Extract metadata
    model_match = re.search(r"Model: (.+)", content)
    dataset_match = re.search(r"Dataset: (.+)", content)
    prompt_match = re.search(r"Prompt Type: (.+)", content)

    if not (model_match and dataset_match and prompt_match):
        return None

    model = model_match.group(1).strip()
    dataset = dataset_match.group(1).strip()
    prompt = prompt_match.group(1).strip()

    # Extract weighted average for overall metrics
    weighted_match = re.search(
        r"weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", content
    )
    if weighted_match:
        weighted_precision = float(weighted_match.group(1))
        weighted_recall = float(weighted_match.group(2))
        weighted_f1 = float(weighted_match.group(3))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    # Extract Inference Time Mean
    # Example: Mean: 1.1926s
    time_match = re.search(r"Mean: (\d+\.\d+)s", content)
    latency = float(time_match.group(1)) if time_match else 0.0

    return {
        "Model": model,
        "Dataset": dataset,
        "Prompt": prompt,
        "Weighted_Precision": weighted_precision,
        "Weighted_Recall": weighted_recall,
        "Weighted_F1": weighted_f1,
        "Latency": latency,
    }


def generate_summary(logs_dir="reports"):
    """Generate summary dataframes from all logs - one for each dataset."""
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    data = []

    for log_file in log_files:
        # Skip training logs or other non-evaluation logs if any
        if "training" in log_file:
            continue

        parsed = parse_log_file(log_file)
        if parsed:
            data.append(parsed)

    if not data:
        print("No valid evaluation logs found.")
        return None, None

    df = pd.DataFrame(data)

    # Normalize dataset names to handle inconsistent capitalization
    df["Dataset"] = df["Dataset"].str.lower()

    # Separate data by dataset
    clariq_df = df[df["Dataset"] == "clariq"].copy()
    ambignq_df = df[df["Dataset"] == "ambignq"].copy()

    def create_dataset_table(dataset_df, dataset_name):
        """Create a comprehensive table for a specific dataset."""
        if dataset_df.empty:
            return pd.DataFrame()

        # Group by Model and Prompt to handle potential duplicates
        # Take the most recent run (last in the list)
        dataset_df = dataset_df.groupby(["Model", "Prompt"]).last().reset_index()

        # Create the table with weighted average metrics
        table = pd.DataFrame()
        table["Model"] = dataset_df["Model"]
        table["Prompt Type"] = dataset_df["Prompt"]

        # Weighted average metrics
        table["Weighted Avg - Precision"] = dataset_df["Weighted_Precision"]
        table["Weighted Avg - Recall"] = dataset_df["Weighted_Recall"]
        table["Weighted Avg - F1"] = dataset_df["Weighted_F1"]

        # Latency - mark as N/A if 0 (e.g., for OpenAI models)
        table["Latency (s)"] = dataset_df["Latency"].apply(
            lambda x: "N/A" if x == 0 else x
        )

        # Sort by Weighted F1 descending
        table = table.sort_values("Weighted Avg - F1", ascending=False)

        return table

    clariq_table = create_dataset_table(clariq_df, "ClariQ")
    ambignq_table = create_dataset_table(ambignq_df, "AmbigNQ")

    return clariq_table, ambignq_table


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation summary table.")
    parser.add_argument(
        "--logs_dir", default="logs", help="Directory containing log files"
    )
    parser.add_argument(
        "--output", default="llms_evals_summary_table.md", help="Output markdown file"
    )
    args = parser.parse_args()

    # Adjust logs_dir path relative to where script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # If logs_dir is relative, make it relative to script location
    if not os.path.isabs(args.logs_dir):
        logs_path = os.path.join(script_dir, args.logs_dir)
        if os.path.exists(logs_path):
            args.logs_dir = logs_path
        elif not os.path.exists(args.logs_dir):
            # If still not found, show error with helpful message
            print(f"Error: Logs directory not found at: {args.logs_dir}")
            print(f"Also tried: {logs_path}")
            print(f"Script location: {script_dir}")
            return

    clariq_table, ambignq_table = generate_summary(args.logs_dir)

    if clariq_table is None and ambignq_table is None:
        return

    # Format for Markdown - Round metrics to 4 decimal places
    def format_table(table_df):
        if table_df.empty:
            return ""
        numeric_cols = table_df.select_dtypes(include=["float"]).columns
        table_df[numeric_cols] = table_df[numeric_cols].round(4)
        return table_df.to_markdown(index=False)

    clariq_markdown = format_table(clariq_table)
    ambignq_markdown = format_table(ambignq_table)

    # Print to console
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - COMPREHENSIVE CLASSIFICATION REPORT")
    print("=" * 80 + "\n")

    if clariq_markdown:
        print("## ClariQ Dataset\n")
        print(clariq_markdown)
        print("\n")

    if ambignq_markdown:
        print("## AmbigNQ Dataset\n")
        print(ambignq_markdown)
        print("\n")

    # Save to file
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        f.write("# Evaluation Summary - Comprehensive Classification Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(
            "This report presents classification performance metrics for ambiguity detection models across two datasets.\n\n"
        )
        f.write("---\n\n")

        if clariq_markdown:
            f.write("## ClariQ Dataset\n\n")
            f.write(clariq_markdown)
            f.write("\n\n")

        if ambignq_markdown:
            f.write("## AmbigNQ Dataset\n\n")
            f.write(ambignq_markdown)
            f.write("\n\n")

        f.write("---\n\n")
        f.write("### Metrics Explanation\n\n")
        f.write(
            "- **Precision**: Proportion of positive identifications that were actually correct\n"
        )
        f.write(
            "- **Recall**: Proportion of actual positives that were identified correctly\n"
        )
        f.write(
            "- **F1**: Harmonic mean of precision and recall (2 * (precision * recall) / (precision + recall))\n"
        )
        f.write(
            "- **Weighted Avg**: Metrics weighted by support (number of samples) for each class\n"
        )
        f.write("- **Latency**: Average inference time per sample in seconds\n")

    print("=" * 80)
    print(f"Summary saved to: {os.path.abspath(args.output)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

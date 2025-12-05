#!/usr/bin/env python3
"""
Generate a markdown table with metrics from evaluation files.
"""

import os
import glob
import re
from collections import defaultdict


def parse_eval_file(eval_file):
    """Extract metrics from .eval file."""
    metrics = {}
    try:
        with open(eval_file, "r") as f:
            for line in f:
                line = line.strip()

                # Check for errors
                if "Traceback" in line or "Error" in line:
                    return None

                # Parse ndcg_cut_10
                if "ndcg_cut_10" in line and "all" in line:
                    parts = line.split()
                    metrics["ndcg@10"] = float(parts[-1])
                # Parse recall_1000 FIRST (before recall_100!)
                elif line.startswith("recall_1000") and "all" in line:
                    parts = line.split()
                    metrics["r@1000"] = float(parts[-1])
                # Parse recall_100 (must come after recall_1000 check)
                elif line.startswith("recall_100") and "all" in line:
                    parts = line.split()
                    metrics["r@100"] = float(parts[-1])
    except Exception as e:
        print(f"Error parsing {eval_file}: {e}")
        return None
    return metrics if metrics else None


def parse_run_filename(filename):
    """Extract method, dataset, prompt_type, and model from run filename."""
    # Remove .txt or .txt.eval
    name = filename.replace(".txt.eval", "").replace(".txt", "")

    # Pattern: run.beir.{METHOD}.{DATASET}.{PROMPT_TYPE}.{MODEL}
    # Use regex to properly handle BM25 methods

    # Try pattern with prompt_type and model
    match = re.match(
        r"run\.beir\.([a-z0-9.-]+)\.([a-z0-9-]+)\.(few-shot|zero-shot|original|ambig2doc)\.(.+)",
        name,
    )
    if match:
        return {
            "method": match.group(1),
            "dataset": match.group(2),
            "prompt_type": match.group(3),
            "model": match.group(4),
        }

    # Try pattern with just prompt_type
    match = re.match(
        r"run\.beir\.([a-z0-9.-]+)\.([a-z0-9-]+)\.(few-shot|zero-shot|original|ambig2doc)",
        name,
    )
    if match:
        return {
            "method": match.group(1),
            "dataset": match.group(2),
            "prompt_type": match.group(3),
            "model": None,
        }

    # Try pattern for original (no prompt_type)
    match = re.match(r"run\.beir\.([a-z0-9.-]+)\.([a-z0-9-]+)$", name)
    if match:
        return {
            "method": match.group(1),
            "dataset": match.group(2),
            "prompt_type": "original",
            "model": None,
        }

    return None


def generate_markdown_table(runs_dir, output_file="metrics_table.md"):
    """Generate a markdown table from all .eval files."""

    eval_files = glob.glob(os.path.join(runs_dir, "*.txt.eval"))

    # Organize data: data[dataset][config][method] = metrics
    data = defaultdict(lambda: defaultdict(dict))

    for eval_file in eval_files:
        filename = os.path.basename(eval_file)
        info = parse_run_filename(filename)

        if not info:
            print(f"Skipping {filename}: Could not parse")
            continue

        metrics = parse_eval_file(eval_file)

        if metrics is None or not metrics:
            print(f"Skipping {filename}: No metrics found or evaluation failed")
            continue

        # Create config key with prompt type and model
        if info["prompt_type"] == "original":
            config_key = f"{info['dataset']}(baseline)"
        else:
            # Abbreviate prompt type
            if info["prompt_type"] == "zero-shot":
                prompt_abbr = "zs"
            elif info["prompt_type"] == "few-shot":
                prompt_abbr = "fs"
            else:  # ambig2doc
                prompt_abbr = "ambig2doc"

            if info["model"]:
                config_key = f"{info['dataset']}-{prompt_abbr}-{info['model']}"
            else:
                config_key = f"{info['dataset']}-{prompt_abbr}"

        # Store by dataset and config
        data[info["dataset"]][config_key][info["method"]] = metrics

    # Generate markdown with wide table format
    with open(output_file, "w") as f:
        f.write("# Retrieval Evaluation Results\n\n")

        # Get all datasets
        datasets = sorted(data.keys())

        for dataset in datasets:
            f.write(f"## {dataset.upper()}\n\n")

            # Write table header
            f.write("|  | **BM25 Flat** | | | **BM25 MF** | | |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            f.write("|  | nDCG@10 | R@100 | R@1000 | nDCG@10 | R@100 | R@1000 |\n")

            # Get all configs for this dataset and sort them
            configs = sorted(
                data[dataset].keys(),
                key=lambda x: ("" if "baseline" in x else "1" + x, x),
            )

            # Write data rows
            for config_key in configs:
                methods_data = data[dataset][config_key]

                # Get metrics for each method
                flat_metrics = methods_data.get("bm25-flat", {})
                mf_metrics = methods_data.get("bm25-multifield", {})

                # Format values
                flat_ndcg = (
                    f"{flat_metrics.get('ndcg@10', 0):.4f}"
                    if "ndcg@10" in flat_metrics
                    else "-"
                )
                flat_r100 = (
                    f"{flat_metrics.get('r@100', 0):.4f}"
                    if "r@100" in flat_metrics
                    else "-"
                )
                flat_r1000 = (
                    f"{flat_metrics.get('r@1000', 0):.4f}"
                    if "r@1000" in flat_metrics
                    else "-"
                )

                mf_ndcg = (
                    f"{mf_metrics.get('ndcg@10', 0):.4f}"
                    if "ndcg@10" in mf_metrics
                    else "-"
                )
                mf_r100 = (
                    f"{mf_metrics.get('r@100', 0):.4f}"
                    if "r@100" in mf_metrics
                    else "-"
                )
                mf_r1000 = (
                    f"{mf_metrics.get('r@1000', 0):.4f}"
                    if "r@1000" in mf_metrics
                    else "-"
                )

                # Write row
                f.write(
                    f"| {config_key} | {flat_ndcg} | {flat_r100} | {flat_r1000} | {mf_ndcg} | {mf_r100} | {mf_r1000} |\n"
                )

            f.write("\n")

    print(f"Metrics table saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate metrics table from evaluation results"
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing run files and .eval files",
    )
    parser.add_argument(
        "--output", default="metrics_table.md", help="Output markdown file"
    )

    args = parser.parse_args()

    generate_markdown_table(args.runs_dir, args.output)

#!/usr/bin/env python3
"""
Run searches across all retrieval methods for all query files.
Supports: BM25 Flat, BM25 Multifield
"""

import os
import glob
import argparse
import subprocess
import re


RETRIEVAL_CONFIGS = {
    "bm25-flat": {
        "module": "pyserini.search.lucene",
        "index_template": "beir-v1.0.0-{dataset}.flat",
        "args": lambda dataset, prompt_type, model_type, query_file, output_file: [
            "python",
            "-m",
            "pyserini.search.lucene",
            "--threads",
            "16",
            "--batch-size",
            "128",
            "--index",
            f"beir-v1.0.0-{dataset}.flat",
            "--topics",
            query_file,
            "--output",
            output_file,
            "--output-format",
            "trec",
            "--hits",
            "1000",
            "--bm25",
            "--remove-query",
        ],
    },
    "bm25-multifield": {
        "module": "pyserini.search.lucene",
        "index_template": "beir-v1.0.0-{dataset}.multifield",
        "args": lambda dataset, prompt_type, model_type, query_file, output_file: [
            "python",
            "-m",
            "pyserini.search.lucene",
            "--threads",
            "16",
            "--batch-size",
            "128",
            "--index",
            f"beir-v1.0.0-{dataset}.multifield",
            "--topics",
            query_file,
            "--output",
            output_file,
            "--output-format",
            "trec",
            "--hits",
            "1000",
            "--bm25",
            "--remove-query",
            "--fields",
            "contents=1.0",
            "title=1.0",
        ],
    },
}


def extract_query_info(filename):
    """Extract dataset, prompt_type, and model_type from query filename."""
    # Pattern 1: topics.beir-v1.0.0-{dataset}.test.tsv (baseline)
    match = re.match(r"topics\.beir-v1\.0\.0-([a-z0-9-]+)\.test\.tsv$", filename)
    if match:
        return {
            "dataset": match.group(1),
            "prompt_type": "original",
            "model_type": None,
        }

    # Pattern 2: clarified_topics.beir-v1.0.0-{dataset}.test_{prompt_type}_{model}_timestamp.tsv
    match = re.match(
        r"clarified_topics\.beir-v1\.0\.0-([a-z0-9-]+)\.test_(zero-shot|few-shot|ambig2doc)_([^_]+)(?:_\d+_\d+)?\.tsv$",
        filename,
    )
    if match:
        return {
            "dataset": match.group(1),
            "prompt_type": match.group(2),
            "model_type": match.group(3),
        }

    return None


def generate_output_filename(retrieval_method, dataset, prompt_type, model_type):
    """Generate consistent output filename."""
    if model_type:
        return f"run.beir.{retrieval_method}.{dataset}.{prompt_type}.{model_type}.txt"
    else:
        return f"run.beir.{retrieval_method}.{dataset}.{prompt_type}.txt"


def main():
    # Restrict to CUDA devices 0, 2, and 3
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

    parser = argparse.ArgumentParser(
        description="Run all retrieval methods on all query files"
    )
    parser.add_argument(
        "--queries-dir", default="queries", help="Directory containing query files"
    )
    parser.add_argument(
        "--output-dir", default="runs", help="Directory for output files"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--retrieval",
        nargs="+",
        choices=list(RETRIEVAL_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Which retrieval methods to run",
    )
    args = parser.parse_args()

    # Convert paths to absolute
    queries_dir = os.path.abspath(args.queries_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(queries_dir):
        print(f"Error: Queries directory '{queries_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Find all query files
    query_files = glob.glob(os.path.join(queries_dir, "*.tsv"))

    if not query_files:
        print(f"No .tsv files found in {queries_dir}")
        return

    # Determine which retrieval methods to run
    if "all" in args.retrieval:
        methods_to_run = ["bm25-flat", "bm25-multifield"]
    else:
        methods_to_run = args.retrieval

    print(f"Found {len(query_files)} query files")
    print(
        f"Running {len(methods_to_run)} retrieval methods: {', '.join(methods_to_run)}\n"
    )

    total_commands = 0
    successful_commands = 0
    failed_commands = 0
    skipped_commands = 0

    for query_file in sorted(query_files):
        filename = os.path.basename(query_file)
        info = extract_query_info(filename)

        if not info:
            print(f"⚠️  Skipping {filename}: Could not parse filename")
            continue

        dataset = info["dataset"]
        prompt_type = info["prompt_type"]
        model_type = info["model_type"]

        print(f"\n{'='*80}")
        print(f"Query File: {filename}")
        print(f"  Dataset: {dataset}")
        print(f"  Prompt Type: {prompt_type}")
        if model_type:
            print(f"  Model Type: {model_type}")
        print(f"{'='*80}")

        for retrieval_method in methods_to_run:
            config = RETRIEVAL_CONFIGS[retrieval_method]
            output_file = os.path.join(
                output_dir,
                generate_output_filename(
                    retrieval_method, dataset, prompt_type, model_type
                ),
            )

            # Skip if output already exists (unless dry-run)
            if os.path.exists(output_file) and not args.dry_run:
                print(f"  ⏭️  {retrieval_method}: Output already exists, skipping")
                skipped_commands += 1
                continue

            # Generate command
            try:
                cmd = config["args"](
                    dataset, prompt_type, model_type, query_file, output_file
                )
            except Exception as e:
                print(f"  ❌ {retrieval_method}: Error generating command: {e}")
                failed_commands += 1
                continue

            total_commands += 1
            cmd_str = " ".join(cmd)

            print(f"\n  🔍 {retrieval_method}")
            print(f"     Output: {os.path.basename(output_file)}")

            if args.dry_run:
                print(f"     Command: {cmd_str}")
                print(f"     [DRY RUN - not executed]")
            else:
                try:
                    result = subprocess.run(
                        cmd, check=True, capture_output=True, text=True
                    )
                    print(f"     ✅ Success")
                    successful_commands += 1
                except subprocess.CalledProcessError as e:
                    print(f"     ❌ Failed: {e}")
                    print(f"     stderr: {e.stderr[:200]}")
                    failed_commands += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total commands: {total_commands}")
    print(f"Successful: {successful_commands}")
    print(f"Failed: {failed_commands}")
    print(f"Skipped: {skipped_commands}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

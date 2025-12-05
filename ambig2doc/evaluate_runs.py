import os
import subprocess
import re
import glob


def get_qrels_key(filename):
    # Extract the dataset part from the filename
    # Example 1: clarified_topics.beir-v1.0.0-scidocs.test_few-shot_llama3.1-8b.txt
    # Target: beir-v1.0.0-scidocs-test
    # Example 2: run.beir.bm25-flat.scidocs.txt
    # Target: beir-v1.0.0-scidocs-test
    # Example 3: run.beir.bm25-multifield.scidocs.few-shot.llama3.1-8b.txt
    # Target: beir-v1.0.0-scidocs-test

    # Try pattern for clarified topics files
    match = re.search(r"(beir-v1\.0\.0-[a-z0-9-]+)\.test", filename)
    if match:
        return match.group(1) + "-test"

    # Try pattern for run files with variants: run.beir.{METHOD}.DATASET.(few-shot|zero-shot|original|ambig2doc)...txt
    # Supports BM25-flat and BM25-multifield methods
    match = re.search(
        r"run\.beir\.[a-z0-9.-]+\.([a-z0-9-]+)\.(few-shot|zero-shot|original|ambig2doc)",
        filename,
    )
    if match:
        dataset = match.group(1)
        return f"beir-v1.0.0-{dataset}-test"

    # Try pattern for baseline files: run.beir.{METHOD}.DATASET.txt
    match = re.search(r"run\.beir\.[a-z0-9.-]+\.([a-z0-9-]+)\.txt", filename)
    if match:
        dataset = match.group(1)
        return f"beir-v1.0.0-{dataset}-test"

    return None


def evaluate_runs(runs_dir):
    run_files = glob.glob(os.path.join(runs_dir, "*.txt"))

    for run_file in run_files:
        filename = os.path.basename(run_file)
        qrels_key = get_qrels_key(filename)

        if not qrels_key:
            print(f"Skipping {filename}: Could not determine qrels key.")
            continue

        # Check if run file is empty
        if os.path.getsize(run_file) == 0:
            print(f"Skipping {filename}: Run file is empty")
            continue

        output_file = run_file + ".eval"

        # Skip if .eval file already exists and is not empty
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            print(
                f"Skipping {filename}: Evaluation already exists at {os.path.basename(output_file)}"
            )
            continue

        print(f"Evaluating {filename} with qrels {qrels_key}...")

        with open(output_file, "w") as f:
            # NDCG@10
            cmd_ndcg = [
                "python",
                "-m",
                "pyserini.eval.trec_eval",
                "-c",
                "-m",
                "ndcg_cut.10",
                qrels_key,
                run_file,
            ]
            f.write(f"Command: {' '.join(cmd_ndcg)}\n")
            f.flush()
            subprocess.run(cmd_ndcg, stdout=f, stderr=subprocess.STDOUT)
            f.write("\n")

            # Recall@100
            cmd_recall_100 = [
                "python",
                "-m",
                "pyserini.eval.trec_eval",
                "-c",
                "-m",
                "recall.100",
                qrels_key,
                run_file,
            ]
            f.write(f"Command: {' '.join(cmd_recall_100)}\n")
            f.flush()
            subprocess.run(cmd_recall_100, stdout=f, stderr=subprocess.STDOUT)
            f.write("\n")

            # Recall@1000
            cmd_recall_1000 = [
                "python",
                "-m",
                "pyserini.eval.trec_eval",
                "-c",
                "-m",
                "recall.1000",
                qrels_key,
                run_file,
            ]
            f.write(f"Command: {' '.join(cmd_recall_1000)}\n")
            f.flush()
            subprocess.run(cmd_recall_1000, stdout=f, stderr=subprocess.STDOUT)
            f.write("\n")

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    runs_directory = "/u40/chanc187/source/pyseerini/runs"
    evaluate_runs(runs_directory)

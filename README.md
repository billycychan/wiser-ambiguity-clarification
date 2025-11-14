# Evaluation Script Guide

## Overview

This script (`scripts/evaluate.py`) evaluates Large Language Models (LLMs) on ambiguity detection tasks. It supports multiple models (Phi-3, Llama-3.1-8B, Llama-3.2-3B) and datasets (currently ClariQ), using zero-shot or few-shot prompting. The script performs inference, computes classification metrics, and saves results to log files and prediction files.

## Prerequisites

- Python 3.x
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - scikit-learn
  - tqdm
  - transformers (or equivalent for LLM pipelines)
- Access to the specified LLMs. Ensure the model pipelines are properly set up in the `core/llms/` directory (e.g., `phi3_mini_128_8b_instruct.py`).
- Preprocessed dataset files in the `data/` directory (e.g., `clariq_preprocessed.tsv`).

## Installation and Setup

1. Ensure you have the project directory structure as shown in the workspace.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Verify that the LLM modules in `core/llms/` are correctly configured with their respective pipelines and formatters. Each module should export a `pipeline` object and a `format_prompt` function.
4. Confirm that dataset files are present in `data/`.

## Usage

Run the script from the project root directory:

```
python scripts/evaluate.py [options]
```

### Command-Line Arguments

| Argument | Type | Default | Choices/Description |
|----------|------|---------|---------------------|
| `--models` | list (space-separated) | all | Models to evaluate. Choices: `Phi-3`, `Llama-3.1-8B`, `Llama-3.2-3B`. |
| `--datasets` | list (space-separated) | all | Datasets to use. Choices: `clariq`. |
| `--prompt_type` | string | `zero_shot` | Type of system prompt. Options: `zero_shot`, `few_shot`, or `custom`. |
| `--max_new_tokens` | int | 5 | Maximum number of new tokens for generation. |
| `--temperature` | float | 0.0 | Temperature for sampling. |
| `--do_sample` | flag | False | Enable sampling (no value needed, just include the flag). |
| `--batch_size` | int | 32 | Batch size for inference. |

### Examples

1. Evaluate all models on the ClariQ dataset with zero-shot prompting:
   ```
   python scripts/evaluate.py
   ```

2. Evaluate only Phi-3 on ClariQ with few-shot prompting and sampling enabled:
   ```
   python scripts/evaluate.py --models Phi-3 --datasets clariq --prompt_type few_shot --do_sample --temperature 0.7
   ```

3. Evaluate Llama-3.1-8B on ClariQ with custom generation settings:
   ```
   python scripts/evaluate.py --models "Llama-3.1-8B" --max_new_tokens 10 --batch_size 16
   ```

## Output

- **Console Output**: Prints evaluation progress and the classification report (precision, recall, F1-score for "Not Ambiguous" and "Ambiguous" classes).
- **Log Files**: Saved in `logs/` with filenames like `phi_3_clariq_20251113_120000.log`. Contains model info, arguments, classification report, and inference time statistics (min, max, mean).
- **Prediction Files**: Saved in `logs/` as TSV files (e.g., `phi_3_clariq_20251113_120000_predictions.tsv`). Includes the original dataset columns plus a `predicted_label` column (0 for not ambiguous, 1 for ambiguous).

## Notes and Troubleshooting

- The script normalizes model outputs by checking for "ambiguous" or "yes" in the first line of the generated text (case-insensitive).
- Ensure the `core/llms/` modules are correctly implemented, as the script imports them directly.
- If you encounter import errors, verify the Python path and module structures.
- For large datasets, adjust `batch_size` to fit your hardware (e.g., reduce for lower memory).
- Inference times are measured per query and averaged across batches.
- The script creates the `logs/` directory if it doesn't exist.
# Synthetic Query Generation with Llama-3.3-70B

This script generates synthetic query data for ambiguity detection using the Llama-3.3-70B-Instruct model via vLLM.

## Overview

The script generates balanced datasets of specific and ambiguous queries across various topics. For each topic, it creates:
- **5 distinct user intents** (actionable or knowledge-seeking goals)
- **2 queries per intent**: one specific (label=0) and one ambiguous (label=1)
- **Total: 10 rows per topic**

## Output Format

The script generates a TSV file with the following columns:
- `topic`: The general topic (e.g., "Activism", "Banking", "Health")
- `initial_request`: The user's query text
- `binary_label`: 0 for specific queries, 1 for ambiguous queries
- `user_information_need`: Description of the user's information need

### Example Output
```tsv
topic	initial_request	binary_label	user_information_need
Activism	What are the main goals of activism?	0	Understand the primary objectives of activism.
Activism	What's the point of activism?	1	Seeking information about activism but intent is unclear.
```

## Files

- **`llama33_70b_instruct.py`**: Main script that runs the generation
- **`prompt.py`**: Contains the system and user prompts with instructions
- **`topics.py`**: List of 100+ topics to generate queries for

## How to Run

### Prerequisites
1. vLLM installed
2. Access to 4 GPUs (configured for tensor parallelism)
3. Access to `nvidia/Llama-3.3-70B-Instruct-FP8` model

### Running the Script

```bash
cd /u40/chanc187/source/wiser_ambiguity_clarification/intent_cf/synthgen
python llama33_70b_instruct.py
```

### Output Files

The script creates two outputs:

1. **TSV File** (main output):
   - Location: `../data/Llama-3.3-70B-Instruct_balanced_strict_<timestamp>.tsv`
   - Format: Tab-separated values with headers
   - Contains: Structured query data ready for training

2. **Log File** (debugging):
   - Location: `logs/llama33_70b_instruct_<timestamp>.log`
   - Format: Raw LLM responses organized by topic
   - Contains: Unprocessed model outputs for verification

## Configuration

### Model Settings (in `llama33_70b_instruct.py`)
- **Model**: `nvidia/Llama-3.3-70B-Instruct-FP8`
- **Tensor Parallel Size**: 4 GPUs
- **GPU Memory Utilization**: 85%
- **Temperature**: 1.0
- **Top-p**: 0.95
- **Max Tokens**: 5000

### Topics
The script processes all 100+ topics defined in `topics.py`. To customize:
1. Edit `topics.py` to add/remove topics
2. Or modify the script to use a subset: `topics[:10]` for first 10 topics

### Prompt Customization
To modify how queries are generated, edit `prompt.py`:
- **`SYSTEM_PROMPT`**: Instructions for the model's behavior
- **`USER_PROMPT_TEMPLATE`**: Instructions for query generation format

## Expected Runtime

- **~100 topics** with current settings
- **10 rows per topic** = ~1000 total rows
- Runtime depends on GPU availability and model loading time

## Troubleshooting

### GPU Memory Issues
If you encounter OOM errors, try:
- Reducing `gpu_memory_utilization` (e.g., 0.75 instead of 0.85)
- Reducing `tensor_parallel_size` (requires fewer GPUs)

### Output Format Issues
If the TSV output is malformed:
1. Check the raw log file to see the model's actual responses
2. The prompt in `prompt.py` enforces strict tab-delimited format
3. Model may occasionally add extra text - these lines will be included

### Missing Dependencies
```bash
pip install vllm tqdm
```

## Example Usage

```python
# Load the generated TSV file
import pandas as pd

df = pd.read_csv('data/Llama-3.3-70B-Instruct_balanced_strict_20251202_212035.tsv', sep='\t')

print(f"Total queries: {len(df)}")
print(f"Specific queries (0): {(df['binary_label'] == 0).sum()}")
print(f"Ambiguous queries (1): {(df['binary_label'] == 1).sum()}")
print(f"Unique topics: {df['topic'].nunique()}")
```

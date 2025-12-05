# LLM Direct Evaluation for Query Ambiguity Detection

## Research Objective

This module evaluates the capability of Large Language Models (LLMs) to **detect ambiguous queries** in conversational systems. Given a user query, the model must classify it as either **ambiguous** or **unambiguous** a binary classification task that is fundamental to improving clarification question generation in dialogue systems.

## Problem Definition

**Task:** Binary classification of query ambiguity  
**Input:** A natural language query (e.g., *"When did he land on the moon?"*)  
**Output:** `yes` (ambiguous) or `no` (unambiguous)

A query is considered **ambiguous** if it contains:
- **Unfamiliar references** — Unknown or mixed entities (*"Find the price of Samsung Chromecast"*)
- **Contradictions** — Logically inconsistent information
- **Lexical ambiguity** — Words with multiple meanings (*"Can you book the book?"*)
- **Semantic ambiguity** — Unclear or implausible meaning
- **Missing context** — Unspecified who/when/where/what (*"When did he land on the moon?"*)

---

## Methodology

### Approach: Zero-Shot and Few-Shot Prompting

We evaluate LLMs using two prompting strategies:

1. **Zero-shot:** The model receives only task instructions without examples
2. **Few-shot:** The model receives task instructions with categorized examples of ambiguity types

### Inference Framework: vLLM

We use [vLLM](https://docs.vllm.ai/) for high-performance inference, achieving **2-5x speedup** over standard Hugging Face Transformers through optimized batching and memory management.

---

## Evaluated Models

| Model Family | Models | Size Range |
|--------------|--------|------------|
| **Llama** | Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B, Llama-3.3-70B | 1B - 70B |
| **Gemma** | Gemma-3-1B, Gemma-3-4B, Gemma-3-27B | 1B - 27B |
| **Phi** | Phi-3-mini-128k | 4B |
| **GPT (OpenAI)** | GPT-4.1, GPT-4.1-mini, GPT-4.1-nano | Proprietary |

---

## Datasets

| Dataset | Description | Size | Source |
|---------|-------------|------|--------|
| **ClariQ** | Conversational queries requiring clarification | ~200 samples | TREC Conversational Assistance |
| **AmbigNQ** | Ambiguous natural questions | ~2,000 samples | Google Research |

**Labels:** Binary (0 = unambiguous, 1 = ambiguous)

---

## Results Summary

### ClariQ Dataset Performance (Weighted F1)

| Rank | Model | Prompt | F1 Score |
|------|-------|--------|----------|
| 1 | Llama-3.2-3B | Zero-shot | **0.826** |
| 2 | Gemma-3-1B | Few-shot | 0.819 |
| 3 | Phi-3 | Zero-shot | 0.819 |
| 4 | Gemma-3-4B | Zero-shot | 0.819 |
| 5 | Llama-3.3-70B | Zero-shot | 0.765 |

### AmbigNQ Dataset Performance (Weighted F1)

| Rank | Model | Prompt | F1 Score |
|------|-------|--------|----------|
| 1 | Llama-3.3-70B | Zero-shot | **0.541** |
| 2 | GPT-4.1-nano | Zero-shot | 0.539 |
| 3 | Gemma-3-27B | Zero-shot | 0.532 |
| 4 | Llama-3.1-8B | Zero-shot | 0.530 |
| 5 | GPT-4.1 | Zero-shot | 0.526 |

### Key Findings

1. **Smaller models can outperform larger ones** on ClariQ — Llama-3.2-3B (3B params) achieved the highest F1 (0.826), outperforming 70B models
2. **Zero-shot generally outperforms few-shot** — Counter-intuitively, providing examples often reduced performance
3. **AmbigNQ is more challenging** — Best F1 scores (~0.54) are significantly lower than ClariQ (~0.83), indicating room for improvement
4. **Inference is fast** — vLLM achieves sub-millisecond latency for most models

---

## Project Structure

```
llm_direct_evaluation/
├── core/
│   ├── prompts.py              # Zero-shot and few-shot prompt templates
│   └── llms/                   # Model-specific configurations
│       ├── llama31_8b_instruct.py
│       ├── llama33_70b_instruct_fp8.py
│       ├── gemma_3_27b_it.py
│       └── ...
├── evaluations/
│   ├── llm_inference/
│   │   ├── evaluate.py         # Main evaluation script
│   │   └── utils.py            # Helper functions
│   └── openai/                 # OpenAI API batch evaluation
├── results/
│   ├── predictions/            # Model predictions (TSV files)
│   ├── reports/                # Evaluation logs
│   └── classification_summary.md
└── scripts/
    └── run_all_evals.sh        # Batch evaluation script
```

---

## How to Run

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- GPU with sufficient memory (8GB+ for small models, 80GB+ for 70B)

### Installation

```bash
# From the repository root
pip install -r requirements.txt
```

### Running Evaluations

```bash
# Basic usage: Single model, single dataset
python evaluations/llm_inference/evaluate.py \
    --models "Llama-3.1-8B" \
    --datasets clariq \
    --prompt_type zero_shot

# Comprehensive evaluation: Multiple models and datasets
python evaluations/llm_inference/evaluate.py \
    --models "Llama-3.1-8B" "Gemma-3-1B" "Llama-3.3-70B" \
    --datasets ambignq clariq \
    --prompt_type few_shot
```

### Command-Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | Models to evaluate (space-separated) | All models |
| `--datasets` | Datasets to use (`ambignq`, `clariq`) | Both |
| `--prompt_type` | `zero_shot` or `few_shot` | `zero_shot` |
| `--batch_size` | Inference batch size | 32 |
| `--seed` | Random seed for reproducibility | 0 |

---

## Output Files

### Prediction Files (`results/predictions/`)
TSV files containing original data plus model predictions:
```
query                           label   predicted_label
When did he land on the moon?   1       1
What is the capital of France?  0       0
```

### Evaluation Reports (`results/reports/`)
Detailed classification metrics including precision, recall, F1, and latency statistics.

---

## Related Work

This module is part of a larger project on **Clarification Need Prediction (CNP)**:
- **[Synthetic Data Generation](../intent_cf/)** — Generate training data using LLMs
- **[PLM Fine-tuning](../intent_cf/plm_training/)** — Train smaller models (DistilBERT, RoBERTa) on synthetic data

---

## References

- [vLLM: Easy, Fast, and Cheap LLM Serving](https://docs.vllm.ai/)
- [AmbigQA: Answering Ambiguous Open-domain Questions](https://nlp.cs.washington.edu/ambigqa/)
- [ClariQ: A Large-Scale and Diverse Dataset for Clarification Question Generation](https://github.com/aliannejadi/ClariQ)

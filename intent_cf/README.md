# Synthetic Data Generation & PLM Fine-tuning for Query Ambiguity Detection

## Research Objective

This module investigates whether **smaller Pre-trained Language Models (PLMs)** can match or exceed larger LLMs on ambiguity detection when trained on **LLM-generated synthetic data**. This approach offers significant cost and latency advantages for production deployment.

## Problem Definition

**Research Question:** Can we distill the ambiguity detection capabilities of large LLMs (70B parameters) into smaller, efficient models (66M-125M parameters)?

**Approach:**
1. **Generate** synthetic training data using large LLMs
2. **Fine-tune** smaller PLMs (DistilBERT, RoBERTa) on the synthetic data
3. **Evaluate** on real-world datasets (ClariQ, AmbigNQ)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA GENERATION                        │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────────┐   │
│  │ 100+     │───▶│ LLM (vLLM)  │───▶│ ~1000 synthetic queries  │   │
│  │ Topics   │    │ Llama/GPT   │    │ (balanced: 50% amb/unambig)│  │
│  └──────────┘    └─────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PLM FINE-TUNING                               │
│  ┌──────────────────────┐    ┌─────────────────────────────────┐   │
│  │ Synthetic Data       │───▶│ DistilBERT / RoBERTa            │   │
│  │ (train)              │    │ 5-fold cross-validation         │   │
│  └──────────────────────┘    └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EVALUATION                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Test on ClariQ (~300 samples) and AmbigNQ (~2000 samples)   │   │
│  │ Metrics: Precision, Recall, F1, Inference Latency           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Methodology

### 1. Synthetic Data Generation

For each of **100+ topics** (e.g., "Banking", "Health", "Travel"), we generate:
- **5 distinct user intents** per topic
- **2 queries per intent**: one specific (label=0), one ambiguous (label=1)
- **Total: ~1000 balanced query pairs**

**Example Generated Data:**
| Topic | Query | Label | User Information Need |
|-------|-------|-------|----------------------|
| Banking | What are the current interest rates for savings accounts at Chase? | 0 | Find specific savings account rates |
| Banking | What about the rates? | 1 | Seeking rate information but context unclear |

### 2. PLM Fine-tuning

We fine-tune two efficient transformer models:

| Model | Parameters | Architecture |
|-------|------------|--------------|
| **DistilBERT** | 66M | 6-layer distilled BERT |
| **RoBERTa** | 125M | Robustly optimized BERT |

**Training Configuration:**
- 5-fold stratified cross-validation
- Early stopping on F1 score
- Class-weighted loss for imbalanced data
- Label smoothing (0.1) for regularization

### 3. Teacher Models (Synthetic Data Sources)

| Model | Size | Provider |
|-------|------|----------|
| Llama-3.1-8B | 8B | Meta |
| Llama-3.3-70B | 70B | Meta (via NVIDIA FP8) |
| Gemma-3-27B | 27B | Google |
| GPT-4.1 | Proprietary | OpenAI |
| GPT-4.1-mini | Proprietary | OpenAI |
| GPT-4.1-nano | Proprietary | OpenAI |

---

## Results Summary

### Best Performing Configurations

#### ClariQ Dataset (Weighted F1)

| Rank | Student Model | Teacher (Synthetic Data) | F1 Score |
|------|---------------|--------------------------|----------|
| 1 | RoBERTa | GPT-4.1 | **0.83** |
| 2 | DistilBERT | Llama-3.3-70B | 0.79 |
| 3 | DistilBERT | GPT-4.1 | 0.66 |
| 4 | DistilBERT | Gemma-3-27B | 0.64 |
| - | *Baseline (no fine-tuning)* | - | 0.11 |

#### AmbigNQ Dataset (Weighted F1)

| Rank | Student Model | Teacher (Synthetic Data) | F1 Score |
|------|---------------|--------------------------|----------|
| 1 | RoBERTa | Llama-3.3-70B | **0.52** |
| 2 | DistilBERT | Llama-3.3-70B | 0.49 |
| 3 | RoBERTa | GPT-4.1 | 0.48 |
| 4 | DistilBERT | GPT-4.1 | 0.43 |
| - | *Baseline (no fine-tuning)* | - | 0.42 |

### Key Findings

1. **Synthetic data significantly improves performance** — Fine-tuned models outperform zero-shot baselines by 6-7x on ClariQ
2. **Llama-3.3-70B produces the best synthetic data** — Consistently yields highest student model performance
3. **RoBERTa slightly outperforms DistilBERT** — The extra parameters (125M vs 66M) provide modest gains
4. **Inference is extremely fast** — ~2ms per sample vs ~17ms for LLM direct inference (8x speedup)
5. **AmbigNQ remains challenging** — Real-world ambiguous questions are harder than conversational queries

---

## Project Structure

```
intent_cf/
├── generate.py                 # Main synthetic data generation script
├── synthgen/
│   ├── prompt.py               # Generation prompt templates
│   ├── topics.py               # 100+ topic categories
│   ├── llama33_70b_instruct.py # Llama-specific generation
│   ├── gemma_3_27b_it.py       # Gemma-specific generation
│   └── openai/                 # OpenAI batch API generation
├── plm_training/
│   ├── distillbert.py          # DistilBERT fine-tuning
│   ├── roberta.py              # RoBERTa fine-tuning
│   └── baselines/              # Zero-shot baseline scripts
└── results/
    └── plm/                    # Training reports & predictions
```

---

## How to Run

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- ~8GB GPU memory for PLM training
- ~80GB GPU memory for Llama-3.3-70B generation (4x GPUs)

### Installation

```bash
# From the repository root
pip install -r requirements.txt
```

### Step 1: Generate Synthetic Data

```bash
# Using Llama-3.3-70B (best quality)
python generate.py --model "Llama-3.3-70B" --output_dir ../data

# Using smaller model (faster)
python generate.py --model "Gemma-3-27B" --output_dir ../data

# Test with limited topics
python generate.py --model "Llama-3.1-8B" --limit 10 --output_dir ../data
```

**Output:** `../data/<model>_balanced_strict.tsv`

### Step 2: Fine-tune PLM

```bash
# Train DistilBERT
cd plm_training
python distillbert.py

# Train RoBERTa
python roberta.py
```

**Note:** Edit `DATASET_NAME` in the script to select which synthetic dataset to use:
```python
DATASET_NAME = "llama3.3-70B"  # Options: llama31_8b, gemma-3-27b, gpt-4-1, etc.
```

### Step 3: Run Baselines (Optional)

```bash
cd plm_training/baselines
python distillbert.py  # Zero-shot baseline
python roberta.py      # Zero-shot baseline
```

---

## Output Files

### Synthetic Data (`../data/`)
TSV files with balanced query pairs:
```
topic    initial_request    binary_label    user_information_need
Banking  What are the current interest rates?    0    Find specific rate information
Banking  What about the rates?    1    Seeking rate info but context unclear
```

### Training Reports (`results/plm/`)
- `*_report_*.txt` — Classification metrics for ClariQ and AmbigNQ
- `*_predictions_*.tsv` — Model predictions on test sets
- `*_training_summary_*.txt` — Training logs and hyperparameters

---

## Configuration Options

### Synthetic Generation (`generate.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | LLM to use for generation | Required |
| `--batch_size` | Topics per batch | 32 |
| `--output_dir` | Output directory | `../data` |
| `--limit` | Limit number of topics | All (100+) |

### PLM Training (`plm_training/*.py`)

Edit constants at the top of each script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 10 | Maximum epochs |
| `per_device_train_batch_size` | 32 | Batch size |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `label_smoothing_factor` | 0.1 | Regularization |

---

## Comparison with Direct LLM Evaluation

| Metric | PLM (DistilBERT) | LLM (Llama-3.3-70B) |
|--------|------------------|---------------------|
| **Parameters** | 66M | 70B |
| **ClariQ F1** | 0.79 | 0.77 |
| **AmbigNQ F1** | 0.49 | 0.54 |
| **Inference Latency** | ~2ms | ~17ms |
| **GPU Memory** | ~1GB | ~80GB (4x GPUs) |
| **Cost** | Low (local) | High (GPU hours) |

**Takeaway:** Fine-tuned PLMs achieve comparable performance to 1000x larger LLMs with 8x faster inference and minimal compute requirements.

---

## Related Work

- **[LLM Direct Evaluation](../llm_direct_evaluation/)** — Evaluate LLMs directly on ambiguity detection
- **[Datasets](../data/)** — Preprocessed ClariQ and AmbigNQ test sets

---

## References

- [DistilBERT: A distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://docs.vllm.ai/)

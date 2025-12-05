# Evaluation Summary - Comprehensive Classification Report

**Generated on:** 2025-12-01 20:24:50

This report presents classification performance metrics for ambiguity detection models across two datasets.

---

## ClariQ Dataset

| Model         | Prompt Type   |   Weighted Avg - Precision |   Weighted Avg - Recall |   Weighted Avg - F1 | Latency (s)   |
|:--------------|:--------------|---------------------------:|------------------------:|--------------------:|:--------------|
| Llama-3.2-3B  | zero_shot     |                     0.8941 |                  0.8796 |              0.8264 | 0.001         |
| Gemma-3-1B    | few_shot      |                     0.7678 |                  0.8763 |              0.8185 | 0.0015        |
| Phi-3         | zero_shot     |                     0.7678 |                  0.8763 |              0.8185 | 0.0019        |
| Gemma-3-4B    | zero_shot     |                     0.7678 |                  0.8763 |              0.8185 | 0.0011        |
| Gemma-3-1B    | zero_shot     |                     0.7675 |                  0.8729 |              0.8168 | 0.0011        |
| Gemma-3-4B    | few_shot      |                     0.7675 |                  0.8729 |              0.8168 | 0.0022        |
| Llama-3.2-1B  | few_shot      |                     0.7667 |                  0.8662 |              0.8134 | 0.0007        |
| Gemma-3-27B   | zero_shot     |                     0.806  |                  0.7759 |              0.7898 | 0.0083        |
| Phi-3         | few_shot      |                     0.8201 |                  0.7492 |              0.7783 | 0.0014        |
| Llama-3.3-70B | zero_shot     |                     0.8316 |                  0.7258 |              0.7651 | 0.0175        |
| gpt-4.1-mini  | zero_shot     |                     0.8461 |                  0.7124 |              0.7577 | N/A           |
| Llama-3.3-70B | few_shot      |                     0.8481 |                  0.699  |              0.7479 | 0.0185        |
| Llama-3.2-3B  | few_shot      |                     0.8318 |                  0.6823 |              0.7336 | 0.0011        |
| Llama-3.1-8B  | zero_shot     |                     0.8145 |                  0.6823 |              0.7314 | 0.0021        |
| gpt-4.1       | zero_shot     |                     0.8441 |                  0.6756 |              0.7296 | N/A           |
| Gemma-3-27B   | few_shot      |                     0.8384 |                  0.6689 |              0.724  | 0.0073        |
| gpt-4.1-mini  | few_shot      |                     0.8567 |                  0.6321 |              0.6949 | N/A           |
| Llama-3.2-1B  | zero_shot     |                     0.7789 |                  0.612  |              0.6755 | 0.0008        |
| gpt-4.1-nano  | zero_shot     |                     0.8133 |                  0.5819 |              0.6532 | N/A           |
| Llama-3.1-8B  | few_shot      |                     0.8495 |                  0.5017 |              0.5761 | 0.0034        |
| gpt-4.1       | few_shot      |                     0.8475 |                  0.4883 |              0.5629 | N/A           |
| gpt-4.1-nano  | few_shot      |                     0.8179 |                  0.3177 |              0.3667 | N/A           |

## AmbigNQ Dataset

| Model         | Prompt Type   |   Weighted Avg - Precision |   Weighted Avg - Recall |   Weighted Avg - F1 | Latency (s)   |
|:--------------|:--------------|---------------------------:|------------------------:|--------------------:|:--------------|
| Llama-3.3-70B | zero_shot     |                     0.5392 |                  0.5435 |              0.5409 | 0.0168        |
| gpt-4.1-nano  | zero_shot     |                     0.545  |                  0.5365 |              0.5393 | N/A           |
| Gemma-3-27B   | zero_shot     |                     0.5299 |                  0.536  |              0.5322 | 0.0094        |
| Llama-3.1-8B  | zero_shot     |                     0.5266 |                  0.5395 |              0.5298 | 0.0017        |
| gpt-4.1       | zero_shot     |                     0.614  |                  0.542  |              0.5257 | N/A           |
| Llama-3.3-70B | few_shot      |                     0.5375 |                  0.5165 |              0.5194 | 0.0185        |
| Phi-3         | few_shot      |                     0.5273 |                  0.5649 |              0.51   | 0.0016        |
| gpt-4.1       | few_shot      |                     0.5668 |                  0.5145 |              0.5039 | N/A           |
| gpt-4.1-mini  | zero_shot     |                     0.5801 |                  0.509  |              0.4864 | N/A           |
| Gemma-3-27B   | few_shot      |                     0.5095 |                  0.478  |              0.4765 | 0.0084        |
| Llama-3.1-8B  | few_shot      |                     0.5055 |                  0.4755 |              0.4748 | 0.0026        |
| Llama-3.2-3B  | few_shot      |                     0.4991 |                  0.4725 |              0.4735 | 0.001         |
| Llama-3.2-1B  | zero_shot     |                     0.5242 |                  0.5794 |              0.4622 | 0.0006        |
| Llama-3.2-1B  | few_shot      |                     0.551  |                  0.5854 |              0.4494 | 0.0008        |
| gpt-4.1-mini  | few_shot      |                     0.5604 |                  0.479  |              0.4383 | N/A           |
| Llama-3.2-3B  | zero_shot     |                     0.527  |                  0.5849 |              0.4357 | 0.0008        |
| Gemma-3-4B    | few_shot      |                     0.4807 |                  0.5839 |              0.4343 | 0.0013        |
| Gemma-3-1B    | zero_shot     |                     0.5501 |                  0.5854 |              0.4341 | 0.0006        |
| Gemma-3-4B    | zero_shot     |                     0.3427 |                  0.5854 |              0.4323 | 0.0011        |
| Gemma-3-1B    | few_shot      |                     0.3427 |                  0.5854 |              0.4323 | 0.0008        |
| Phi-3         | zero_shot     |                     0.3426 |                  0.5849 |              0.4321 | 0.0013        |
| gpt-4.1-nano  | few_shot      |                     0.5389 |                  0.4505 |              0.3811 | N/A           |

---

### Metrics Explanation

- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1**: Harmonic mean of precision and recall (2 * (precision * recall) / (precision + recall))
- **Weighted Avg**: Metrics weighted by support (number of samples) for each class
- **Latency**: Average inference time per sample in seconds

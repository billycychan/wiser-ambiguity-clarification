# Evaluation Commands for All Model-Dataset-Prompt Combinations
# Total: 8 models × 2 datasets × 2 prompt types = 32 commands

## Gemma Models

### Gemma-3-1B
```bash
python scripts/evaluate.py --models "Gemma-3-1B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-1B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-1B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-1B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

### Gemma-3-4B
```bash
python scripts/evaluate.py --models "Gemma-3-4B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-4B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-4B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-4B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

### Gemma-3-27B
```bash
python scripts/evaluate.py --models "Gemma-3-27B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-27B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-27B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Gemma-3-27B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

## Llama Models

### Llama-3.1-8B
```bash
python scripts/evaluate.py --models "Llama-3.1-8B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.1-8B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.1-8B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.1-8B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

### Llama-3.2-1B
```bash
python scripts/evaluate.py --models "Llama-3.2-1B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.2-1B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.2-1B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.2-1B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

### Llama-3.2-3B
```bash
python scripts/evaluate.py --models "Llama-3.2-3B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.2-3B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.2-3B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.2-3B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

### Llama-3.3-70B
```bash
python scripts/evaluate.py --models "Llama-3.3-70B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.3-70B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.3-70B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Llama-3.3-70B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

## Phi Model

### Phi-3
```bash
python scripts/evaluate.py --models "Phi-3" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Phi-3" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Phi-3" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
python scripts/evaluate.py --models "Phi-3" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
```

## Quick Run Scripts

### Run All Commands Sequentially
```bash
# Copy and paste all commands above, or use this loop
for model in "Gemma-3-1B" "Gemma-3-4B" "Gemma-3-27B" "Llama-3.1-8B" "Llama-3.2-1B" "Llama-3.2-3B" "Llama-3.3-70B" "Phi-3"; do
  for dataset in ambignq clariq; do
    for prompt in zero_shot few_shot; do
      echo "Running: $model + $dataset + $prompt"
      python scripts/evaluate.py --models "$model" --datasets "$dataset" --prompt_type "$prompt" --batch_size 64 --seed 0
    done
  done
done
```

### Run with GPU Assignment (for large models)
```bash
# For Llama-3.3-70B (requires multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --models "Llama-3.3-70B" --datasets ambignq --prompt_type zero_shot --batch_size 64 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --models "Llama-3.3-70B" --datasets ambignq --prompt_type few_shot --batch_size 64 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --models "Llama-3.3-70B" --datasets clariq --prompt_type zero_shot --batch_size 64 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --models "Llama-3.3-70B" --datasets clariq --prompt_type few_shot --batch_size 64 --seed 0
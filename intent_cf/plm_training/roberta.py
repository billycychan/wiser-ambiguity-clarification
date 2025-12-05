import pandas as pd
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
import logging
import os
import time
from datetime import datetime

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Switch between datasets by changing the DATASET_NAME value
# Options: 'llama31_8b' or 'gemma-3-27b' or 'llama3.3-70B'
DATASET_NAME = "gpt-4-1-nano"  # Change this to switch datasets

# Updated paths to use relative path from intent_cf/plm_training/
DATA_DIR = "../../data"
DATASET_PATHS = {
    "llama31_8b": f"{DATA_DIR}/llama31_8b_balanced_strict.tsv",
    "gemma-3-27b": f"{DATA_DIR}/gemma-3-27b-it_balanced_strict.tsv",
    "llama3.3-70B": f"{DATA_DIR}/Llama-3.3-70B-Instruct_balanced_strict.tsv",
    "gpt-4-1": f"{DATA_DIR}/gpt_4_1_balanced_strict.tsv",
    "gpt-4-1-mini": f"{DATA_DIR}/gpt_4_1_mini_balanced_strict.tsv",
    "gpt-4-1-nano": f"{DATA_DIR}/gpt_4_1_nano_balanced_strict.tsv",
}

# Get the selected dataset path
TRAIN_DATA_PATH = DATASET_PATHS[DATASET_NAME]
# ============================================================================


def get_model_and_config():
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=2,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=config
    )
    return model, config


def get_training_args():
    return TrainingArguments(
        output_dir="../logs/roberta",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        logging_dir="../logs/roberta",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=2,
        label_smoothing_factor=0.1,
        max_grad_norm=1.0,
    )


def get_optimizer_and_scheduler(
    model, dataset_len, num_train_epochs, batch_size, grad_accum
):
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.02)
    effective_batch_size = batch_size * grad_accum
    num_training_steps = (dataset_len // effective_batch_size) * num_train_epochs
    if num_training_steps <= 0:
        num_training_steps = 1
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def custom_loss(model, inputs, return_outputs=False, class_weights=None):
    outputs = model(**inputs)
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(outputs.logits.device)
    )
    loss = loss_fct(
        outputs.logits.view(-1, model.config.num_labels), inputs["labels"].view(-1)
    )
    return (loss, outputs) if return_outputs else loss


class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            message = f"Epoch {state.epoch}: {metrics}"
            print(message)
            logging.info(message)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class WeightedLossTrainer(Trainer):
    """Trainer that applies class weighting to the loss function."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            # Handle DataParallel wrapped models
            num_labels = (
                model.module.config.num_labels
                if hasattr(model, "module")
                else model.config.num_labels
            )
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, dtype=torch.float).to(
                    logits.device
                )
            )
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            # default HF loss
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure logs directory exists and set up logging
    os.makedirs("../logs/roberta", exist_ok=True)
    logging.basicConfig(
        filename=f"../logs/roberta/roberta_{DATASET_NAME}_{timestamp}_training.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    # Load data
    train_df = pd.read_csv(
        TRAIN_DATA_PATH,
        sep="\t",
    )
    clariq_df = pd.read_csv(f"{DATA_DIR}/clariq_preprocessed.tsv", sep="\t")
    ambignq_df = pd.read_csv(f"{DATA_DIR}/ambignq_preprocessed.tsv", sep="\t")

    train_texts = train_df["initial_request"].tolist()
    train_labels = train_df["binary_label"].tolist()
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    val_texts = clariq_df["initial_request"].tolist()
    val_labels = clariq_df["binary_label"].tolist()
    ambignq_texts = ambignq_df["initial_request"].tolist()
    ambignq_labels = ambignq_df["binary_label"].tolist()

    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    ambignq_dataset = TextDataset(ambignq_texts, ambignq_labels, tokenizer)

    # Cross-validation for evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_train_dataset = TextDataset(fold_train_texts, fold_train_labels, tokenizer)

        model, _ = get_model_and_config()
        training_args = get_training_args()
        optimizer, scheduler = get_optimizer_and_scheduler(
            model,
            len(fold_train_dataset),
            training_args.num_train_epochs,
            training_args.per_device_train_batch_size,
            training_args.gradient_accumulation_steps,
        )

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=val_dataset,  # Use real validation set
            data_collator=DataCollatorWithPadding(tokenizer),
            optimizers=(optimizer, scheduler),
            compute_metrics=lambda p: {
                "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
                "f1": f1_score(
                    p.label_ids, p.predictions.argmax(-1), average="weighted"
                ),
                "precision": precision_score(
                    p.label_ids, p.predictions.argmax(-1), average="weighted"
                ),
                "recall": recall_score(
                    p.label_ids, p.predictions.argmax(-1), average="weighted"
                ),
            },
            callbacks=[
                MetricsCallback(),
                EarlyStoppingCallback(early_stopping_patience=3),
            ],
            class_weights=class_weights,
        )

        trainer.train()
        best_f1 = trainer.state.best_metric
        cv_f1_scores.append(best_f1)
        print(f"Fold {fold+1} Best F1: {best_f1}")
        logging.info(f"Fold {fold+1} Best F1: {best_f1}")

    print(f"CV Average Best F1: {sum(cv_f1_scores)/len(cv_f1_scores)}")
    logging.info(f"CV Average Best F1: {sum(cv_f1_scores)/len(cv_f1_scores)}")

    # Final Model
    model, _ = get_model_and_config()
    training_args = get_training_args()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        len(train_dataset),
        training_args.num_train_epochs,
        training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps,
    )
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=ambignq_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(optimizer, scheduler),
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
            "f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="weighted"),
            "precision": precision_score(
                p.label_ids, p.predictions.argmax(-1), average="weighted"
            ),
            "recall": recall_score(
                p.label_ids, p.predictions.argmax(-1), average="weighted"
            ),
        },
        callbacks=[MetricsCallback(), EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )
    trainer.train()

    # Evaluate on ClariQ
    start_time = time.time()
    predictions = trainer.predict(val_dataset)
    inference_time = time.time() - start_time
    num_samples = len(val_dataset)
    avg_inference_time = inference_time / num_samples
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # Classification report for ClariQ
    report = classification_report(
        labels, preds, target_names=["not ambiguous", "ambiguous"]
    )
    accuracy = accuracy_score(labels, preds)

    print("Classification Report for ClariQ:")
    print(report)
    print(f"Validation Accuracy: {accuracy}")
    print(f"Inference Time: {inference_time} seconds")
    print(f"Average Inference Time per Sample: {avg_inference_time} seconds")

    # Log to file
    logging.info("Final Classification Report for ClariQ:\n" + report)
    logging.info(f"Final Validation Accuracy: {accuracy}")
    logging.info(f"Inference Time: {inference_time} seconds")
    logging.info(f"Average Inference Time per Sample: {avg_inference_time} seconds")

    # Evaluate on AmbigNQ
    start_time_ambignq = time.time()
    predictions_ambignq = trainer.predict(ambignq_dataset)
    inference_time_ambignq = time.time() - start_time_ambignq
    num_samples_ambignq = len(ambignq_dataset)
    avg_inference_time_ambignq = inference_time_ambignq / num_samples_ambignq
    preds_ambignq = predictions_ambignq.predictions.argmax(-1)
    labels_ambignq = predictions_ambignq.label_ids

    # Classification report for AmbigNQ
    report_ambignq = classification_report(
        labels_ambignq, preds_ambignq, target_names=["not ambiguous", "ambiguous"]
    )
    accuracy_ambignq = accuracy_score(labels_ambignq, preds_ambignq)

    print("Classification Report for AmbigNQ:")
    print(report_ambignq)
    print(f"Validation Accuracy: {accuracy_ambignq}")
    print(f"Inference Time: {inference_time_ambignq} seconds")
    print(f"Average Inference Time per Sample: {avg_inference_time_ambignq} seconds")

    # Log to file
    logging.info("Final Classification Report for AmbigNQ:\n" + report_ambignq)
    logging.info(f"Final Validation Accuracy: {accuracy_ambignq}")
    logging.info(f"Inference Time: {inference_time_ambignq} seconds")
    logging.info(
        f"Average Inference Time per Sample: {avg_inference_time_ambignq} seconds"
    )

    # Save to logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(
        f"../logs/roberta/roberta_{DATASET_NAME}_report_{timestamp}.txt", "w"
    ) as f:
        f.write("Classification Report for ClariQ:\n")
        f.write(report)
        f.write(f"\nValidation Accuracy: {accuracy}\n")
        f.write(f"Inference Time: {inference_time} seconds\n")
        f.write(f"Average Inference Time per Sample: {avg_inference_time} seconds\n\n")
        f.write("Classification Report for AmbigNQ:\n")
        f.write(report_ambignq)
        f.write(f"\nValidation Accuracy: {accuracy_ambignq}\n")
        f.write(f"Inference Time: {inference_time_ambignq} seconds\n")
        f.write(
            f"Average Inference Time per Sample: {avg_inference_time_ambignq} seconds\n"
        )

    # Save training summary (basic metrics)
    training_summary = (
        f"Training completed with {training_args.num_train_epochs} epochs.\n"
        f"Final Validation Accuracy on ClariQ: {accuracy}\n"
        f"Inference Time on ClariQ: {inference_time} seconds\n"
        f"Average Inference Time per Sample on ClariQ: {avg_inference_time} seconds\n"
        f"Final Validation Accuracy on AmbigNQ: {accuracy_ambignq}\n"
        f"Inference Time on AmbigNQ: {inference_time_ambignq} seconds\n"
        f"Average Inference Time per Sample on AmbigNQ: {avg_inference_time_ambignq} seconds\n"
    )
    with open(
        f"../logs/roberta/roberta_{DATASET_NAME}_training_summary_{timestamp}.txt", "w"
    ) as f:
        f.write(training_summary)

    # Save predictions
    # ClariQ predictions
    clariq_df = pd.DataFrame(
        {"initial_request": val_texts, "binary_label": val_labels, "prediction": preds}
    )
    clariq_df.to_csv(
        f"../logs/roberta/roberta_{DATASET_NAME}_clariq_{timestamp}_predictions.tsv",
        sep="\t",
        index=False,
    )

    # AmbigNQ predictions
    ambignq_df = pd.DataFrame(
        {
            "initial_request": ambignq_texts,
            "binary_label": ambignq_labels,
            "prediction": preds_ambignq,
        }
    )
    ambignq_df.to_csv(
        f"../logs/roberta/roberta_{DATASET_NAME}_ambignq_{timestamp}_predictions.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()

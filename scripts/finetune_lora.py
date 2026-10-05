"""
scripts/finetune_lora.py
========================
LoRA fine-tuning of RoBERTa for sentiment classification.

This is OPTIONAL.  The base app works without it.  Use this when
you have enough labelled in-domain data (≥500 comments) and want
to close the domain gap further.

To enable the fine-tuned model in the app, set in .env:
    USE_LORA_ADAPTER=true

Usage:
    # On local CPU (slow but works):
    python scripts/finetune_lora.py --data data/raw_dataset.csv --epochs 3

    # On Colab (recommended for speed):
    # Upload this file and your dataset, then run the same command.
    # Colab free tier has enough RAM for this config.

LoRA config rationale:
  - r=8        : rank of the adaptation matrices (small = fewer params)
  - alpha=32   : scaling factor (effective scale = alpha/r = 4)
  - dropout=0.1: regularization on LoRA layers
  - targets     : query and value projection layers in each attention head
  - Trainable params: ~0.48% of total (≈600K out of 125M)

Outputs:
  - downloads/lora_adapter/  (the adapter weights, loadable with PEFT)
  - reports/lora_before_after.json  (macro-F1 before and after fine-tuning)

Requirements (not in base requirements.txt):
    pip install peft accelerate
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID     = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ADAPTER_DIR  = os.path.join("downloads", "lora_adapter")
REPORTS_DIR  = "reports"
LABEL2ID     = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL     = {v: k for k, v in LABEL2ID.items()}


def check_dependencies():
    missing = []
    for pkg in ["peft", "accelerate", "torch", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(
            "Missing packages: %s\n"
            "Install with: pip install %s",
            ", ".join(missing), " ".join(missing),
        )
        sys.exit(1)


def load_data(path: str) -> tuple[list[str], list[int]]:
    """
    Load labelled comments from CSV.
    Returns (texts, int_labels).
    """
    from modules.training_data import TRAINING_DATA

    rows = []

    if path and os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            text  = str(row.get("text", "")).strip()
            human = str(row.get("human_label", "")).strip()
            robot = str(row.get("roberta_label", "")).strip()
            label = human if human in LABEL2ID else robot
            if label in LABEL2ID and text:
                rows.append((text, LABEL2ID[label]))

    # Append curated dataset as auxiliary
    for t, l in TRAINING_DATA:
        cap = l.capitalize()
        if cap in LABEL2ID:
            rows.append((t, LABEL2ID[cap]))

    logger.info("Total samples: %d | Distribution: %s",
                len(rows), dict(Counter(r[1] for r in rows)))

    texts  = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    return texts, labels


def evaluate_model(model, tokenizer, texts, labels, device, batch_size=16) -> float:
    """Return macro-F1 on the given split."""
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(batch, truncation=True, max_length=128,
                          padding=True, return_tensors="pt")
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)

    return round(f1_score(labels, all_preds, average="macro", zero_division=0) * 100, 2)


def main():
    check_dependencies()

    parser = argparse.ArgumentParser(description="LoRA fine-tune RoBERTa for sentiment")
    parser.add_argument("--data",       default="data/raw_dataset.csv")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len",    type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import Dataset
    from sklearn.model_selection import train_test_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    texts, labels = load_data(args.data)

    if len(texts) < 30:
        logger.error("Need at least 30 samples for fine-tuning. Got %d.", len(texts))
        sys.exit(1)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Load tokenizer + base model
    logger.info("Loading base model: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Baseline F1 BEFORE fine-tuning
    logger.info("Measuring BEFORE fine-tuning F1…")
    base_model.to(device)
    f1_before = evaluate_model(base_model, tokenizer, X_te, y_te, device)
    logger.info("BEFORE macro-F1: %.2f%%", f1_before)

    # Attach LoRA adapters
    lora_config = LoraConfig(
        task_type      = TaskType.SEQ_CLS,
        r              = 8,           # rank
        lora_alpha     = 32,          # α/r = 4 (effective scaling)
        lora_dropout   = 0.1,
        target_modules = ["query", "value"],   # adapt attention projections
        bias           = "none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # should show ~0.48%

    # Build HuggingFace Dataset objects
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
        )

    train_ds = Dataset.from_dict({"text": X_tr, "labels": y_tr}).map(tokenize, batched=True)
    eval_ds  = Dataset.from_dict({"text": X_te, "labels": y_te}).map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    # TrainingArguments — CPU-safe settings
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir              = ADAPTER_DIR,
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = args.lr,
        weight_decay            = 0.01,
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        logging_steps           = 10,
        no_cuda                 = (device.type == "cpu"),
        report_to               = "none",   # no W&B / tensorboard
        fp16                    = (device.type == "cuda"),
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
    )

    logger.info("Starting LoRA fine-tuning for %d epochs…", args.epochs)
    trainer.train()

    # Save adapter
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    logger.info("Adapter saved to %s", ADAPTER_DIR)

    # F1 AFTER
    logger.info("Measuring AFTER fine-tuning F1…")
    f1_after = evaluate_model(model, tokenizer, X_te, y_te, device)
    logger.info("AFTER  macro-F1: %.2f%%", f1_after)

    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "base_model":     MODEL_ID,
        "lora_r":         8,
        "lora_alpha":     32,
        "lora_target":    ["query", "value"],
        "epochs":         args.epochs,
        "train_samples":  len(X_tr),
        "test_samples":   len(X_te),
        "f1_before":      f1_before,
        "f1_after":       f1_after,
        "improvement":    round(f1_after - f1_before, 2),
    }
    report_path = os.path.join(REPORTS_DIR, "lora_before_after.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n{'='*50}")
    print(f"  LoRA fine-tuning complete")
    print(f"  Macro-F1 BEFORE : {f1_before:.2f}%")
    print(f"  Macro-F1 AFTER  : {f1_after:.2f}%")
    print(f"  Improvement     : {f1_after - f1_before:+.2f}%")
    print(f"  Adapter saved to: {ADAPTER_DIR}/")
    print(f"  Report saved to : {report_path}")
    print(f"{'='*50}\n")
    print("To use the fine-tuned model in the app:")
    print("  Add  USE_LORA_ADAPTER=true  to your .env file")
    print("  Restart the Flask app\n")


if __name__ == "__main__":
    main()

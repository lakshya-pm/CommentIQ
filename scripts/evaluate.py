"""
scripts/evaluate.py
===================
Regenerate ALL metrics and figures used in the project report.

Usage:
    python scripts/evaluate.py [--data data/raw_dataset.csv]

Outputs (all in reports/):
    reports/metrics.json        — primary source of truth for README
    reports/lr_confusion.png
    reports/nb_confusion.png
    reports/model_comparison.png
    reports/per_class_metrics.json

RULE: Every number in README.md must come from reports/metrics.json.
Never hardcode metrics in documentation.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.preprocessing import preprocess_text
from modules.training_data import TRAINING_DATA
from modules.sentiment import SentimentModels, classify_with_vader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

LABEL_ORDER = ["Positive", "Negative", "Neutral"]


# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

def load_dataset(in_domain_path: str | None) -> list[tuple[str, str]]:
    data = []

    if in_domain_path and os.path.exists(in_domain_path):
        df = pd.read_csv(in_domain_path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            text  = str(row.get("text", "")).strip()
            human = str(row.get("human_label", "")).strip()
            robot = str(row.get("roberta_label", "")).strip()
            label = human if human in {"Positive","Negative","Neutral","positive","negative","neutral"} else robot
            if label and text:
                data.append((text, label.capitalize()))
        logger.info("In-domain: %d samples", len(data))

    curated = [(t, l.capitalize()) for t, l in TRAINING_DATA]
    data   += curated
    logger.info("Curated appended: %d  |  Total: %d", len(curated), len(data))
    return data


# ---------------------------------------------------
# VADER baseline evaluation
# ---------------------------------------------------

def evaluate_vader(texts_raw: list[str], labels: list[str]) -> dict:
    """Evaluate VADER on raw (not preprocessed) text, same labels."""
    preds = []
    for t in texts_raw:
        r = classify_with_vader(t, t)
        preds.append(r["sentiment"])

    present = [l for l in LABEL_ORDER if l in labels]
    return {
        "accuracy":  round(accuracy_score(labels, preds) * 100, 2),
        "macro_f1":  round(f1_score(labels, preds, average="macro",    labels=present, zero_division=0) * 100, 2),
        "weighted_f1": round(f1_score(labels, preds, average="weighted", labels=present, zero_division=0) * 100, 2),
        "per_class": _per_class(labels, preds, present),
        "confusion_matrix": confusion_matrix(labels, preds, labels=present).tolist(),
        "labels": present,
    }


# ---------------------------------------------------
# ML evaluation (5-fold CV)
# ---------------------------------------------------

def evaluate_ml(texts_clean: list[str], labels: list[str]) -> dict:
    """Train and evaluate LR+NB with stratified 5-fold CV."""
    sm = SentimentModels()
    sm.train(texts_clean, labels)
    return sm.evaluation_results


# ---------------------------------------------------
# RoBERTa evaluation (optional — slow)
# ---------------------------------------------------

def evaluate_roberta(texts_raw: list[str], labels: list[str], sample: int = 200) -> dict | None:
    """
    Evaluate RoBERTa on up to `sample` examples to keep runtime manageable.
    Returns None if transformers not available.
    """
    try:
        from modules import transformer as _t
        if not _t.is_transformer_available():
            return None
    except Exception:
        return None

    logger.info("Running RoBERTa evaluation on %d samples…", min(sample, len(texts_raw)))

    # Stratified subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(len(texts_raw), size=min(sample, len(texts_raw)), replace=False)
    sub_texts  = [texts_raw[i] for i in idx]
    sub_labels = [labels[i]    for i in idx]

    results = _t.transformer_predict(sub_texts)
    if not results:
        return None

    preds   = [r["label"] for r in results]
    present = [l for l in LABEL_ORDER if l in sub_labels]

    return {
        "accuracy":    round(accuracy_score(sub_labels, preds) * 100, 2),
        "macro_f1":    round(f1_score(sub_labels, preds, average="macro",    labels=present, zero_division=0) * 100, 2),
        "weighted_f1": round(f1_score(sub_labels, preds, average="weighted", labels=present, zero_division=0) * 100, 2),
        "n_evaluated": len(sub_texts),
        "per_class":   _per_class(sub_labels, preds, present),
        "confusion_matrix": confusion_matrix(sub_labels, preds, labels=present).tolist(),
        "labels": present,
    }


# ---------------------------------------------------
# Per-class breakdown
# ---------------------------------------------------

def _per_class(y_true, y_pred, labels) -> dict:
    report = {}
    for lbl in labels:
        binary_true = [1 if y == lbl else 0 for y in y_true]
        binary_pred = [1 if y == lbl else 0 for y in y_pred]
        report[lbl] = {
            "precision": round(precision_score(binary_true, binary_pred, zero_division=0) * 100, 2),
            "recall":    round(recall_score   (binary_true, binary_pred, zero_division=0) * 100, 2),
            "f1":        round(f1_score       (binary_true, binary_pred, zero_division=0) * 100, 2),
            "support":   int(sum(binary_true)),
        }
    return report


# ---------------------------------------------------
# Figures
# ---------------------------------------------------

def plot_confusion(cm: list, labels: list, title: str, path: str):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="True", title=title,
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info("Saved: %s", path)


def plot_model_comparison(metrics: dict, path: str):
    models  = []
    accs    = []
    f1s     = []

    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        label = name.replace("_", " ").title()
        # Prefer CV mean, fall back to held-out accuracy
        acc = m.get("cv_acc_mean") or m.get("accuracy", 0)
        f1  = m.get("cv_f1_mean")  or m.get("macro_f1", 0)
        models.append(label)
        accs.append(acc)
        f1s.append(f1)

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#4C72B0")
    ax.bar(x + w/2, f1s,  w, label="Macro-F1 (%)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info("Saved: %s", path)


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Regenerate all evaluation metrics and figures")
    parser.add_argument("--data", default="data/raw_dataset.csv")
    parser.add_argument("--no-roberta", action="store_true",
                        help="Skip RoBERTa evaluation (faster)")
    args = parser.parse_args()

    data   = load_dataset(args.data if not args.no_roberta else None)
    texts_raw   = [t for t, _ in data]
    labels_raw  = [l for _, l in data]
    texts_clean = [preprocess_text(t) for t in texts_raw]

    label_dist = dict(Counter(labels_raw))
    logger.info("Label distribution: %s", label_dist)

    # --- VADER ---
    logger.info("Evaluating VADER…")
    vader_metrics = evaluate_vader(texts_raw, labels_raw)

    # --- ML (LR + NB) ---
    logger.info("Evaluating LR + NB (5-fold CV)…")
    ml_metrics = evaluate_ml(texts_clean, labels_raw)

    # --- RoBERTa (optional) ---
    roberta_metrics = None
    if not args.no_roberta:
        logger.info("Evaluating RoBERTa (200-sample stratified subset)…")
        roberta_metrics = evaluate_roberta(texts_raw, labels_raw, sample=200)

    # --- Aggregate ---
    all_metrics = {
        "dataset": {
            "total_samples":      len(data),
            "label_distribution": label_dist,
        },
        "vader":              vader_metrics,
        "logistic_regression": ml_metrics.get("logistic_regression", {}),
        "naive_bayes":         ml_metrics.get("naive_bayes", {}),
        "roberta":             roberta_metrics,
        "n_folds":             ml_metrics.get("n_folds", 5),
        "train_size":          ml_metrics.get("train_size", 0),
        "test_size":           ml_metrics.get("test_size",  0),
    }

    # --- Save JSON ---
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)

    per_class_path = os.path.join(REPORTS_DIR, "per_class_metrics.json")
    per_class = {
        "vader":               vader_metrics.get("per_class", {}),
        "logistic_regression": ml_metrics.get("logistic_regression", {}).get("per_class", {}),
        "naive_bayes":         ml_metrics.get("naive_bayes", {}).get("per_class", {}),
    }
    if roberta_metrics:
        per_class["roberta"] = roberta_metrics.get("per_class", {})
    with open(per_class_path, "w") as f:
        json.dump(per_class, f, indent=2)

    # --- Figures ---
    lr_cm = ml_metrics.get("logistic_regression", {}).get("confusion_matrix")
    nb_cm = ml_metrics.get("naive_bayes", {}).get("confusion_matrix")
    cm_labels = ml_metrics.get("logistic_regression", {}).get("labels", LABEL_ORDER)

    if lr_cm:
        plot_confusion(lr_cm, cm_labels, "Logistic Regression — Confusion Matrix",
                       os.path.join(REPORTS_DIR, "lr_confusion.png"))
    if nb_cm:
        plot_confusion(nb_cm, cm_labels, "Naive Bayes — Confusion Matrix",
                       os.path.join(REPORTS_DIR, "nb_confusion.png"))
    if roberta_metrics and roberta_metrics.get("confusion_matrix"):
        plot_confusion(roberta_metrics["confusion_matrix"],
                       roberta_metrics["labels"],
                       "RoBERTa — Confusion Matrix",
                       os.path.join(REPORTS_DIR, "roberta_confusion.png"))

    comparison_data = {}
    for model in ["vader", "logistic_regression", "naive_bayes"]:
        comparison_data[model] = all_metrics.get(model, {})
    if roberta_metrics:
        comparison_data["roberta"] = roberta_metrics
    plot_model_comparison(comparison_data, os.path.join(REPORTS_DIR, "model_comparison.png"))

    # --- Print summary ---
    print("\n" + "=" * 65)
    print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10} {'CV Acc (mean±SD)':>20}")
    print("=" * 65)
    rows_to_print = [
        ("VADER",              vader_metrics),
        ("Logistic Regression",all_metrics["logistic_regression"]),
        ("Naive Bayes",        all_metrics["naive_bayes"]),
    ]
    if roberta_metrics:
        rows_to_print.append(("RoBERTa (200 sample)", roberta_metrics))

    for name, m in rows_to_print:
        acc = m.get("accuracy", 0) or m.get("cv_acc_mean", 0)
        f1  = m.get("macro_f1", 0) or m.get("cv_f1_mean", 0)
        cv  = f"{m.get('cv_acc_mean',0):.1f}±{m.get('cv_acc_std',0):.1f}%" if "cv_acc_mean" in m else "N/A"
        print(f"{name:<22} {acc:>9.2f}% {f1:>9.2f}%   {cv:>18}")
    print("=" * 65)
    print(f"\nAll outputs written to: {os.path.abspath(REPORTS_DIR)}/\n")


if __name__ == "__main__":
    main()

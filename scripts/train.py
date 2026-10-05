"""
scripts/train.py
================
Retrain LR and NB on the in-domain dataset from build_dataset.py.

Usage:
    python scripts/train.py [--data data/raw_dataset.csv] [--folds 5]

If data/raw_dataset.csv exists, it is used as primary training data.
The 462-sample curated set in modules/training_data.py is ALWAYS
concatenated as a clean auxiliary set to prevent accuracy collapse
when the in-domain set is small.

Outputs:
  - downloads/models/logistic_regression.pkl  (overwritten)
  - downloads/models/naive_bayes.pkl          (overwritten)
  - downloads/models/tfidf_vectorizer.pkl     (overwritten)
  - downloads/models/training_metrics.pkl     (overwritten)
  - Prints full metrics table to console
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.preprocessing import preprocess_text
from modules.training_data import TRAINING_DATA
from modules.sentiment import SentimentModels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Canonical label set
VALID_LABELS = {"Positive", "Negative", "Neutral", "positive", "negative", "neutral"}


def load_in_domain(path: str) -> list[tuple[str, str]]:
    """
    Load the human-reviewed CSV from build_dataset.py.

    Uses 'human_label' if filled in, otherwise falls back to 'roberta_label'.
    Skips rows where neither label is valid.
    """
    df     = pd.read_csv(path, encoding="utf-8-sig")
    rows   = []
    skipped = 0

    for _, row in df.iterrows():
        text  = str(row.get("text", "")).strip()
        human = str(row.get("human_label", "")).strip()
        robot = str(row.get("roberta_label", "")).strip()

        label = human if human in VALID_LABELS else robot
        if label not in VALID_LABELS or not text:
            skipped += 1
            continue

        rows.append((text, label.capitalize()))

    logger.info(
        "In-domain dataset: %d usable rows, %d skipped (bad/empty label)", len(rows), skipped
    )
    return rows


def load_curated() -> list[tuple[str, str]]:
    """Load the 462-sample curated set from modules/training_data.py."""
    return [(t, l.capitalize()) for t, l in TRAINING_DATA]


def print_metrics_table(metrics: dict):
    """Pretty-print evaluation results."""
    header = f"{'Model':<25} {'CV Acc':>8} {'±':>4} {'CV F1':>8} {'±':>4} {'Held Acc':>10} {'Macro F1':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for model_name, m in metrics.items():
        if not isinstance(m, dict) or "cv_acc_mean" not in m:
            continue
        print(
            f"{model_name:<25} "
            f"{m.get('cv_acc_mean', 0):>7.2f}% "
            f"{m.get('cv_acc_std',  0):>4.2f} "
            f"{m.get('cv_f1_mean',  0):>7.2f}% "
            f"{m.get('cv_f1_std',   0):>4.2f} "
            f"{m.get('accuracy',    0):>9.2f}% "
            f"{m.get('macro_f1',    0):>9.2f}%"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Train LR+NB sentiment models")
    parser.add_argument(
        "--data",
        default="data/raw_dataset.csv",
        help="Path to in-domain dataset CSV (default: data/raw_dataset.csv)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5)",
    )
    parser.add_argument(
        "--no-curated",
        action="store_true",
        help="Do NOT append the curated 462-sample set (not recommended)",
    )
    args = parser.parse_args()

    # Load data
    data = []

    if os.path.exists(args.data):
        data += load_in_domain(args.data)
        logger.info("Loaded %d in-domain samples", len(data))
    else:
        logger.info(
            "No in-domain dataset found at '%s'. Using curated set only.\n"
            "Run 'python scripts/build_dataset.py' to build one.", args.data
        )

    if not args.no_curated:
        curated = load_curated()
        data   += curated
        logger.info("Appended %d curated samples (total: %d)", len(curated), len(data))

    if len(data) < 30:
        logger.error("Need at least 30 samples to train. Found %d.", len(data))
        sys.exit(1)

    texts  = [preprocess_text(t) for t, _ in data]
    labels = [l for _, l in data]

    from collections import Counter
    logger.info("Label distribution: %s", dict(Counter(labels)))

    # Train and evaluate
    sm = SentimentModels()
    sm.train(texts, labels)

    if not sm.is_trained:
        logger.error("Training failed.")
        sys.exit(1)

    print_metrics_table(sm.evaluation_results)

    logger.info(
        "\nModels saved to downloads/models/\n"
        "Restart the Flask app to pick up the new models."
    )


if __name__ == "__main__":
    main()

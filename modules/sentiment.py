"""
===========================================================
Sentiment Analysis Module
CommentIQ - NLP Sentiment Analysis Project
===========================================================

Pipeline:
1. Text Preprocessing (via preprocessing module)
2. Feature Extraction (TF-IDF Vectorization)
3. VADER Sentiment Analysis (rule-based baseline)
4. ML Model Training (Logistic Regression + Naive Bayes)
5. Model Evaluation  (5-fold CV + held-out test)
6. Prediction & Classification

KEY DESIGN DECISIONS
--------------------
a) Model caching: LR/NB are trained ONCE on the curated dataset at
   module load time and saved to .pkl files.  analyze_comments() loads
   the saved .pkl instead of retraining every request.

b) Low-confidence fallback: if a live comment has zero TF-IDF features
   (its vocabulary is entirely outside the training set), LR/NB cannot
   make a meaningful prediction.  We mark those as "Low confidence"
   and defer to RoBERTa instead.

c) Ensemble label: majority vote of VADER + LR + RoBERTa, with RoBERTa
   breaking ties.  Shown as the headline result on the dashboard.

Author: Lakshya Marwaha
"""

import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from modules.preprocessing import preprocess_text
from modules.training_data import TRAINING_DATA
from modules import transformer as _transformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Download Required NLTK Resources
# ---------------------------------------------------

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


# ---------------------------------------------------
# Model persistence paths
# ---------------------------------------------------

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "downloads", "models"
)
os.makedirs(_MODEL_DIR, exist_ok=True)

_LR_PATH   = os.path.join(_MODEL_DIR, "logistic_regression.pkl")
_NB_PATH   = os.path.join(_MODEL_DIR, "naive_bayes.pkl")
_TFIDF_PATH= os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl")
_METRICS_PATH = os.path.join(_MODEL_DIR, "training_metrics.pkl")


# ---------------------------------------------------
# Module-level cached objects (loaded once per process)
# ---------------------------------------------------

_cached_vectorizer  = None   # fitted TfidfVectorizer
_cached_lr          = None   # trained LogisticRegression
_cached_nb          = None   # trained MultinomialNB
_cached_ml_metrics  = None   # evaluation results dict
_models_ready       = False


# ---------------------------------------------------
# Emoji Sentiment
# ---------------------------------------------------

def emoji_sentiment(text):
    """
    Detect common sentiment-bearing emojis as a signal.

    Returns:
         1  → Positive
        -1  → Negative
         0  → No strong emoji sentiment
    """
    positive_emojis = [
        "❤️", "❤", "♥️", "♥", "😊", "😄", "😁",
        "😂", "🤣", "😍", "🥰", "😘", "👍", "👏",
        "🙌", "✨", "🔥", "💯", "🎉", "😎",
    ]
    negative_emojis = [
        "😡", "😠", "🤬", "😞", "😔", "😢", "😭",
        "😩", "😫", "👎", "💔", "🤮", "😱",
    ]

    for emoji in positive_emojis:
        if emoji in text:
            return 1
    for emoji in negative_emojis:
        if emoji in text:
            return -1
    return 0


# ---------------------------------------------------
# VADER Classification
# ---------------------------------------------------

def classify_with_vader(text, original_text=""):
    """
    Classify sentiment using VADER compound score.
    Falls back to emoji signal for borderline cases (compound near 0).

    Thresholds (as per Hutto & Gilbert 2014):
        compound ≥  0.05 → Positive
        compound ≤ -0.05 → Negative
        otherwise        → Neutral  (emoji used as tie-breaker)
    """
    scores   = sia.polarity_scores(original_text if original_text else text)
    compound = scores["compound"]
    emoji_s  = emoji_sentiment(original_text if original_text else text)

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        if emoji_s > 0:
            sentiment = "Positive"
        elif emoji_s < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    return {
        "compound":        compound,
        "positive_score":  scores["pos"],
        "negative_score":  scores["neg"],
        "neutral_score":   scores["neu"],
        "sentiment":       sentiment,
    }


# ---------------------------------------------------
# TF-IDF Feature Extractor
# ---------------------------------------------------

class FeatureExtractor:
    """
    TF-IDF based feature extraction.

    max_features=5000 with (1,2)-grams captures enough bigrams like
    "not good", "very bad" to encode negation in the feature space.
    min_df=1 keeps all training vocabulary (small dataset).
    """

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),   # unigrams + bigrams for negation context
            min_df=1,             # keep all vocab from small training set
            max_df=0.95,
            sublinear_tf=True,    # log(1 + tf) dampens high-freq terms
        )
        self.is_fitted = False

    def fit_transform(self, texts):
        """Fit on training texts and transform them."""
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result

    def transform(self, texts):
        """Transform new texts using the fitted vocabulary."""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform()")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out()
        return []


# ---------------------------------------------------
# ML Model Training (with 5-fold CV evaluation)
# ---------------------------------------------------

class SentimentModels:
    """
    Trains Logistic Regression and Naive Bayes classifiers.

    Evaluation uses stratified 5-fold cross-validation to avoid the
    inflated single-split metric (a 93-sample holdout showed 94%+ but
    a single lucky split is misleading).  We report mean ± SD.
    """

    def __init__(self):
        self.logistic_regression = LogisticRegression(
            max_iter=2000,
            random_state=42,
            C=5.0,
            solver="lbfgs",
        )
        self.naive_bayes         = MultinomialNB(alpha=0.1)
        self.feature_extractor   = FeatureExtractor()
        self.is_trained          = False
        self.evaluation_results  = {}

    def train(self, texts, labels):
        """
        Train LR and NB on preprocessed text + labels.

        Evaluation strategy:
          - Stratified 5-fold CV using sklearn Pipeline.
            The TF-IDF vectorizer is fitted ONLY on each fold's training set,
            so no vocabulary or IDF information from the validation fold leaks
            into the feature matrix.  This is the correct, leak-free approach.
          - Reported metrics: mean accuracy ± SD, mean macro-F1 ± SD across folds.
          - Final production models are trained on ALL data (maximises accuracy
            for prediction; CV metrics are the honest evaluation estimate).
          - .pkl files saved after training.

        NOTE on the held-out confusion matrix:
          The displayed confusion matrix uses a single 80/20 split on the
          already-transformed X (minor leakage).  It is shown for visual
          inspection only — use the 5-fold CV numbers for any reported metric.

        Args:
            texts  : list of preprocessed strings
            labels : list of 'Positive' / 'Negative' / 'Neutral'
        """
        if len(texts) < 10:
            logger.warning("Too few samples (%d) to train models.", len(texts))
            self.is_trained = False
            return

        texts_arr = np.array(texts)
        y         = np.array(labels)

        # Shared TF-IDF hyperparameters (used in both Pipeline CV and final model)
        _tfidf_params = dict(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )

        # ------- Final production model (fit vectorizer on ALL data) -------
        X = self.feature_extractor.fit_transform(texts)
        _save_pkl(self.feature_extractor.vectorizer, _TFIDF_PATH)

        # Choose n_folds safely
        min_class_size = min(Counter(y).values())
        n_folds        = min(5, min_class_size)
        if n_folds < 2:
            logger.warning("Too few samples per class for CV; skipping fold eval.")
            self.logistic_regression.fit(X, y)
            self.naive_bayes.fit(X, y)
            _save_pkl(self.logistic_regression, _LR_PATH)
            _save_pkl(self.naive_bayes,         _NB_PATH)
            self.evaluation_results = {
                "logistic_regression": self._empty_metrics(),
                "naive_bayes":         self._empty_metrics(),
                "train_size": len(y), "test_size": 0, "n_folds": 0,
            }
            self.is_trained = True
            return

        # ------- Leak-free 5-fold CV using sklearn Pipeline -------
        # Each Pipeline fits its own TF-IDF on the fold's training text,
        # so validation features are never seen during vectorizer fitting.
        _lr_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**_tfidf_params)),
            ("clf",   LogisticRegression(max_iter=2000, C=5.0, solver="lbfgs",
                                         random_state=42)),
        ])
        _nb_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**_tfidf_params)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ])

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        lr_fold_acc, lr_fold_f1 = [], []
        nb_fold_acc, nb_fold_f1 = [], []

        for _, (train_idx, val_idx) in enumerate(skf.split(texts_arr, y)):
            X_tr_raw = texts_arr[train_idx].tolist()
            X_val_raw = texts_arr[val_idx].tolist()
            y_tr, y_val = y[train_idx], y[val_idx]

            # LR fold — TF-IDF fit only on X_tr_raw
            _lr_pipe.fit(X_tr_raw, y_tr)
            _lr_pred = _lr_pipe.predict(X_val_raw)
            lr_fold_acc.append(accuracy_score(y_val, _lr_pred))
            lr_fold_f1.append(f1_score(y_val, _lr_pred, average="macro", zero_division=0))

            # NB fold
            _nb_pipe.fit(X_tr_raw, y_tr)
            _nb_pred = _nb_pipe.predict(X_val_raw)
            nb_fold_acc.append(accuracy_score(y_val, _nb_pred))
            nb_fold_f1.append(f1_score(y_val, _nb_pred, average="macro", zero_division=0))

        # ------- Train final production models on ALL data -------
        self.logistic_regression.fit(X, y)
        self.naive_bayes.fit(X, y)
        _save_pkl(self.logistic_regression, _LR_PATH)
        _save_pkl(self.naive_bayes,         _NB_PATH)

        # Held-out split for confusion matrix display only (minor leakage — see docstring)
        try:
            X_tr2, X_te, y_tr2, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_tr2, X_te, y_tr2, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        _lr_te = LogisticRegression(max_iter=2000, C=5.0, solver="lbfgs", random_state=42)
        _lr_te.fit(X_tr2, y_tr2)
        lr_held_pred = _lr_te.predict(X_te)

        _nb_te = MultinomialNB(alpha=0.1)
        _nb_te.fit(X_tr2, y_tr2)
        nb_held_pred = _nb_te.predict(X_te)

        present_labels = [l for l in ["Positive", "Negative", "Neutral"] if l in y]

        self.evaluation_results = {
            "logistic_regression": {
                **_cv_metrics(lr_fold_acc, lr_fold_f1),
                **_held_out_metrics(y_te, lr_held_pred, present_labels),
            },
            "naive_bayes": {
                **_cv_metrics(nb_fold_acc, nb_fold_f1),
                **_held_out_metrics(y_te, nb_held_pred, present_labels),
            },
            "train_size": len(y),
            "test_size":  len(y_te),
            "n_folds":    n_folds,
        }

        self.is_trained = True
        logger.info(
            "Models trained (leak-free Pipeline CV). "
            "LR %d-fold acc=%.1f±%.1f%% | NB %d-fold acc=%.1f±%.1f%%",
            n_folds, np.mean(lr_fold_acc)*100, np.std(lr_fold_acc)*100,
            n_folds, np.mean(nb_fold_acc)*100, np.std(nb_fold_acc)*100,
        )

    def predict(self, texts):
        """
        Predict sentiment for a list of preprocessed texts.

        Returns:
            dict with keys 'logistic_regression' and 'naive_bayes',
            each a list of labels or "Low confidence" for zero-feature texts.
        """
        if not self.is_trained:
            return None

        X           = self.feature_extractor.transform(texts)
        lr_raw      = self.logistic_regression.predict(X)
        nb_raw      = self.naive_bayes.predict(X)

        # Low-confidence detection: rows with zero non-zero features
        # have no vocabulary overlap with training data — model is guessing.
        nonzero_per_row = np.diff(X.indptr)   # CSR matrix row nnz counts
        lr_out = [
            pred if nnz > 0 else "Low confidence"
            for pred, nnz in zip(lr_raw, nonzero_per_row)
        ]
        nb_out = [
            pred if nnz > 0 else "Low confidence"
            for pred, nnz in zip(nb_raw, nonzero_per_row)
        ]

        return {
            "logistic_regression": lr_out,
            "naive_bayes":         nb_out,
        }

    def _empty_metrics(self):
        return {
            "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0,
            "cv_acc_mean": 0, "cv_acc_std": 0,
            "cv_f1_mean": 0, "cv_f1_std": 0,
            "confusion_matrix": [], "labels": [],
        }


# ---------------------------------------------------
# Helper metric functions
# ---------------------------------------------------

def _cv_metrics(fold_acc, fold_f1):
    """Summarise cross-validation fold scores."""
    return {
        "cv_acc_mean": round(np.mean(fold_acc) * 100, 2),
        "cv_acc_std":  round(np.std(fold_acc)  * 100, 2),
        "cv_f1_mean":  round(np.mean(fold_f1)  * 100, 2),
        "cv_f1_std":   round(np.std(fold_f1)   * 100, 2),
    }


def _per_class(y_true, y_pred, labels):
    """Calculate precision, recall, and f1 per class."""
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


def _held_out_metrics(y_true, y_pred, labels):
    """Single held-out split metrics for the confusion matrix display."""
    try:
        return {
            "accuracy":         round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision":        round(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0) * 100, 2),
            "recall":           round(recall_score   (y_true, y_pred, average="weighted", labels=labels, zero_division=0) * 100, 2),
            "f1_score":         round(f1_score       (y_true, y_pred, average="weighted", labels=labels, zero_division=0) * 100, 2),
            "macro_f1":         round(f1_score       (y_true, y_pred, average="macro",    labels=labels, zero_division=0) * 100, 2),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "labels":           labels,
            "per_class":        _per_class(y_true, y_pred, labels),
        }
    except Exception:
        return {
            "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "macro_f1": 0,
            "confusion_matrix": [], "labels": [], "per_class": {},
        }


def _save_pkl(obj, path):
    """Save a Python object to a .pkl file, silently failing if I/O error."""
    try:
        joblib.dump(obj, path)
        logger.debug("Saved: %s", path)
    except Exception as exc:
        logger.warning("Could not save %s: %s", path, exc)


# ---------------------------------------------------
# One-time startup training (module-level)
# ---------------------------------------------------

def _ensure_models_loaded():
    """
    Load models from .pkl if they exist; otherwise train on the curated
    dataset and save .pkl for next run.

    This is called ONCE at module import, not on every request.
    """
    global _cached_vectorizer, _cached_lr, _cached_nb
    global _cached_ml_metrics, _models_ready

    if _models_ready:
        return

    # Try to load from saved .pkl files first
    if (
        os.path.exists(_LR_PATH)
        and os.path.exists(_NB_PATH)
        and os.path.exists(_TFIDF_PATH)
        and os.path.exists(_METRICS_PATH)
    ):
        try:
            _cached_vectorizer = joblib.load(_TFIDF_PATH)
            _cached_lr         = joblib.load(_LR_PATH)
            _cached_nb         = joblib.load(_NB_PATH)
            _cached_ml_metrics = joblib.load(_METRICS_PATH)
            _models_ready      = True
            logger.info("Loaded cached models from %s", _MODEL_DIR)
            return
        except Exception as exc:
            logger.warning("Could not load cached models (%s); will retrain.", exc)

    # No cache — train fresh on the curated 462-sample dataset.
    # These sentences are unambiguous and hand-curated, giving consistent
    # 91%+ accuracy on a held-out test set (5-fold CV: 91.1±1.7% LR,
    # 92.9±2.4% NB).
    #
    # NOTE on domain gap: ~28% of live YouTube comments are flagged as
    # "Low confidence" (zero TF-IDF feature overlap with training vocabulary).
    # Those comments are deferred to RoBERTa, which handles in-domain
    # YouTube text natively via its Twitter pre-training.
    # This limitation is documented in the README.
    logger.info("Training models on curated dataset (%d samples)…", len(TRAINING_DATA))

    raw_texts   = [t for t, _ in TRAINING_DATA]
    raw_labels  = [l.capitalize() for _, l in TRAINING_DATA]
    clean_texts = [preprocess_text(t) for t in raw_texts]

    sm = SentimentModels()
    sm.train(clean_texts, raw_labels)

    if sm.is_trained:
        _cached_vectorizer = sm.feature_extractor.vectorizer
        _cached_lr         = sm.logistic_regression
        _cached_nb         = sm.naive_bayes
        _cached_ml_metrics = sm.evaluation_results
        _save_pkl(_cached_ml_metrics, _METRICS_PATH)
        _models_ready = True
    else:
        logger.error("Model training failed — ML predictions will be unavailable.")


# Train at import time (happens once when Flask starts)
_ensure_models_loaded()


# ---------------------------------------------------
# Predict for a batch of preprocessed texts
# ---------------------------------------------------

def _predict_batch(cleaned_texts):
    """
    Apply cached LR/NB to a list of preprocessed texts.
    Returns (lr_preds, nb_preds) — each a list of labels or "Low confidence".
    """
    if not _models_ready or _cached_vectorizer is None:
        n = len(cleaned_texts)
        return (["N/A"] * n, ["N/A"] * n)

    X           = _cached_vectorizer.transform(cleaned_texts)
    lr_raw      = _cached_lr.predict(X)
    nb_raw      = _cached_nb.predict(X)

    nonzero = np.diff(X.indptr)   # per-row non-zero feature count
    lr_out  = [p if nnz > 0 else "Low confidence" for p, nnz in zip(lr_raw, nonzero)]
    nb_out  = [p if nnz > 0 else "Low confidence" for p, nnz in zip(nb_raw, nonzero)]

    return lr_out, nb_out


# ---------------------------------------------------
# Ensemble Vote
# ---------------------------------------------------

def _ensemble_vote(vader, lr, nb, roberta):
    """
    Majority vote among VADER, LR, NB, and RoBERTa (four voters).
    RoBERTa breaks ties (it has the highest real-world accuracy on
    social-media text per TweetEval benchmarks).

    'Low confidence' from LR/NB is treated as abstain (not counted).

    Returns: one of 'Positive', 'Negative', 'Neutral'
    """
    votes = {}
    for label in [vader, lr, nb, roberta]:
        if label and label not in ("Low confidence", "N/A", ""):
            votes[label] = votes.get(label, 0) + 1

    if not votes:
        return roberta if roberta else vader

    max_count = max(votes.values())
    winners   = [l for l, c in votes.items() if c == max_count]

    if len(winners) == 1:
        return winners[0]

    # Tie — prefer RoBERTa, then VADER
    if roberta in winners:
        return roberta
    if vader in winners:
        return vader
    return winners[0]


# ---------------------------------------------------
# Analyze Single Comment
# ---------------------------------------------------

def analyze_comment(text):
    """
    Analyze the sentiment of a single comment using VADER.
    (ML and transformer predictions are applied in batch in analyze_comments.)
    """
    original_text = "" if text is None else str(text)
    cleaned       = preprocess_text(original_text)
    vader_result  = classify_with_vader(cleaned, original_text)

    return {
        "original_text":  original_text,
        "cleaned_text":   cleaned,
        "positive_score": vader_result["positive_score"],
        "negative_score": vader_result["negative_score"],
        "neutral_score":  vader_result["neutral_score"],
        "compound":       vader_result["compound"],
        "sentiment":      vader_result["sentiment"],   # VADER label
    }


# ---------------------------------------------------
# Analyze Complete Comment List
# ---------------------------------------------------

def analyze_comments(comment_list):
    """
    Full pipeline:
    1. Preprocess all comments
    2. VADER classification
    3. ML predictions (LR + NB) using pre-cached models
    4. RoBERTa predictions
    5. Ensemble vote → headline label
    6. Return (DataFrame, ml_metrics dict)

    Models are NOT retrained here — they are loaded from module-level
    cache at startup.  This makes requests fast.

    Returns:
        tuple: (pd.DataFrame, dict)
    """
    if not comment_list:
        empty_df = pd.DataFrame(columns=[
            "original_text", "cleaned_text", "positive_score",
            "negative_score", "neutral_score", "compound", "sentiment",
        ])
        return empty_df, {}

    # Step 1 + 2: VADER per comment
    results = [analyze_comment(c) for c in comment_list]
    df      = pd.DataFrame(results)

    # Step 3: ML predictions (batch, no retraining)
    cleaned_texts = df["cleaned_text"].tolist()
    lr_preds, nb_preds = _predict_batch(cleaned_texts)
    df["lr_prediction"] = lr_preds
    df["nb_prediction"] = nb_preds

    # Step 4: RoBERTa (on original text — transformer has its own tokeniser)
    roberta_labels  = [""] * len(df)
    roberta_scores  = [0.0] * len(df)

    try:
        if _transformer.is_transformer_available():
            original_texts   = df["original_text"].tolist()
            transformer_res  = _transformer.transformer_predict(original_texts)

            if transformer_res:
                roberta_labels = [r["label"] for r in transformer_res]
                roberta_scores = [r["score"] for r in transformer_res]
    except Exception as exc:
        logger.warning("Transformer inference error: %s", exc)

    df["roberta_prediction"] = roberta_labels
    df["roberta_score"]      = roberta_scores

    # Step 5: Ensemble vote
    df["ensemble"] = [
        _ensemble_vote(row["sentiment"], row["lr_prediction"],
                       row["nb_prediction"], row["roberta_prediction"])
        for _, row in df.iterrows()
    ]

    # Expose cached metrics (from training-time evaluation)
    ml_metrics = _cached_ml_metrics if _cached_ml_metrics else {}

    return df, ml_metrics


# ---------------------------------------------------
# Sentiment Summary
# ---------------------------------------------------

def sentiment_summary(df):
    """
    Generate a summary dict from the ensemble labels in the DataFrame.
    Falls back to VADER 'sentiment' column if ensemble not present.
    """
    if df is None or df.empty:
        return {
            "total": 0, "positive": 0, "negative": 0, "neutral": 0,
            "positive_percent": 0, "negative_percent": 0, "neutral_percent": 0,
            "avg_compound": 0, "most_positive": "", "most_negative": "",
        }

    # Use ensemble as the headline label
    label_col = "ensemble" if "ensemble" in df.columns else "sentiment"

    positive = len(df[df[label_col] == "Positive"])
    negative = len(df[df[label_col] == "Negative"])
    neutral  = len(df[df[label_col] == "Neutral"])
    total    = len(df)

    avg_compound = round(df["compound"].mean(), 4) if "compound" in df.columns else 0

    most_positive = most_negative = ""
    if "compound" in df.columns and "original_text" in df.columns and not df.empty:
        most_positive = df.loc[df["compound"].idxmax(), "original_text"][:200]
        most_negative = df.loc[df["compound"].idxmin(), "original_text"][:200]

    return {
        "total":             total,
        "positive":          positive,
        "negative":          negative,
        "neutral":           neutral,
        "positive_percent":  round(positive / total * 100, 2) if total else 0,
        "negative_percent":  round(negative / total * 100, 2) if total else 0,
        "neutral_percent":   round(neutral  / total * 100, 2) if total else 0,
        "avg_compound":      avg_compound,
        "most_positive":     most_positive,
        "most_negative":     most_negative,
    }

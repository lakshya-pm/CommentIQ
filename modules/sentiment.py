"""
===========================================================
Sentiment Analysis Module
CommentIQ - NLP Sentiment Analysis Project
===========================================================

Implements the complete ML-based sentiment analysis pipeline:

Pipeline:
1. Text Preprocessing (via preprocessing module)
2. Feature Extraction (TF-IDF)
3. VADER Sentiment Analysis (primary)
4. ML Model Training (Logistic Regression + Naive Bayes)
5. Model Evaluation (Accuracy, Precision, Recall, F1)
6. Prediction & Classification

Features:
✔ Full NLP preprocessing pipeline
✔ TF-IDF Feature Extraction
✔ VADER Sentiment Analysis (primary classifier)
✔ Logistic Regression (secondary ML classifier)
✔ Naive Bayes (secondary ML classifier)
✔ Model evaluation metrics
✔ Emoji-aware sentiment handling
✔ Positive / Neutral / Negative Classification
✔ Compound Sentiment Score
✔ DataFrame Generation
✔ Sentiment Summary with statistics

Author: Lakshya Marwaha
"""

import re
import pandas as pd
import numpy as np
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import os
import joblib

from modules.preprocessing import preprocess_text
from modules.training_data import TRAINING_DATA
from modules import transformer as _transformer


# ---------------------------------------------------
# Download Required Resources
# ---------------------------------------------------

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


sia = SentimentIntensityAnalyzer()


# ---------------------------------------------------
# Emoji Sentiment
# ---------------------------------------------------

def emoji_sentiment(text):
    """
    Detects common sentiment-related emojis.

    Returns:
        1  -> Positive
       -1  -> Negative
        0  -> No strong emoji sentiment
    """

    positive_emojis = [
        "❤️", "❤", "♥️", "♥", "😊", "😄", "😁",
        "😂", "🤣", "😍", "🥰", "😘", "👍", "👏",
        "🙌", "✨", "🔥", "💯", "🎉", "😎"
    ]

    negative_emojis = [
        "😡", "😠", "🤬", "😞", "😔", "😢", "😭",
        "😩", "😫", "👎", "💔", "🤮", "😱"
    ]

    for emoji in positive_emojis:
        if emoji in text:
            return 1

    for emoji in negative_emojis:
        if emoji in text:
            return -1

    return 0


# ---------------------------------------------------
# VADER Sentiment Classification
# ---------------------------------------------------

def classify_with_vader(text, original_text=""):
    """
    Classify sentiment using VADER compound score.
    Uses emoji fallback for borderline cases.
    """

    scores = sia.polarity_scores(
        original_text if original_text else text
    )

    compound = scores["compound"]
    emoji_score = emoji_sentiment(
        original_text if original_text else text
    )

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        if emoji_score > 0:
            sentiment = "Positive"
        elif emoji_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    return {
        "compound": compound,
        "positive_score": scores["pos"],
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "sentiment": sentiment
    }


# ---------------------------------------------------
# TF-IDF Feature Extraction
# ---------------------------------------------------

class FeatureExtractor:
    """
    TF-IDF based feature extraction.

    Converts preprocessed text into numerical feature
    vectors suitable for ML model training.
    """

    def __init__(self, max_features=5000):

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.is_fitted = False

    def fit_transform(self, texts):
        """Fit vectorizer and transform texts."""

        self.is_fitted = True

        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """Transform new texts using fitted vectorizer."""

        if not self.is_fitted:
            return self.fit_transform(texts)

        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """Get feature names (vocabulary)."""

        if self.is_fitted:
            return self.vectorizer.get_feature_names_out()

        return []


# ---------------------------------------------------
# ML Model Training
# ---------------------------------------------------

class SentimentModels:
    """
    Trains Logistic Regression and Naive Bayes classifiers
    on VADER-labeled data for improved classification.
    """

    def __init__(self):

        self.logistic_regression = LogisticRegression(
            max_iter=2000,
            random_state=42,
            C=5.0,
            solver='lbfgs'
        )

        self.naive_bayes = MultinomialNB(
            alpha=0.1
        )

        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.evaluation_results = {}

    def train(self, texts, labels):
        """
        Train both ML models on preprocessed text data.
        Saves trained models to .pkl files after training.

        Args:
            texts: List of preprocessed text strings
            labels: List of sentiment labels
        """

        if len(texts) < 10:
            self.is_trained = False
            return

        # Model save directory
        _model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "downloads", "models"
        )
        os.makedirs(_model_dir, exist_ok=True)

        # Feature extraction
        X = self.feature_extractor.fit_transform(texts)
        y = np.array(labels)

        # Save TF-IDF vectorizer
        try:
            joblib.dump(
                self.feature_extractor.vectorizer,
                os.path.join(_model_dir, "tfidf_vectorizer.pkl")
            )
        except Exception:
            pass

        # Train/test split for evaluation
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        except ValueError:
            # If stratify fails (too few samples per class)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

        # Train Logistic Regression
        try:
            self.logistic_regression.fit(X_train, y_train)
            lr_predictions = self.logistic_regression.predict(X_test)
            lr_metrics = self._evaluate(y_test, lr_predictions)
            # Save model
            joblib.dump(
                self.logistic_regression,
                os.path.join(_model_dir, "logistic_regression.pkl")
            )
        except Exception:
            lr_metrics = self._empty_metrics()

        # Train Naive Bayes
        try:
            self.naive_bayes.fit(X_train, y_train)
            nb_predictions = self.naive_bayes.predict(X_test)
            nb_metrics = self._evaluate(y_test, nb_predictions)
            # Save model
            joblib.dump(
                self.naive_bayes,
                os.path.join(_model_dir, "naive_bayes.pkl")
            )
        except Exception:
            nb_metrics = self._empty_metrics()

        self.evaluation_results = {
            "logistic_regression": lr_metrics,
            "naive_bayes": nb_metrics,
            "test_size": len(y_test),
            "train_size": len(y_train)
        }

        self.is_trained = True

    def predict(self, texts):
        """
        Predict sentiment using trained ML models.

        Returns dict with predictions from both models.
        """

        if not self.is_trained:
            return None

        X = self.feature_extractor.transform(texts)

        lr_predictions = self.logistic_regression.predict(X)
        nb_predictions = self.naive_bayes.predict(X)

        return {
            "logistic_regression": lr_predictions.tolist(),
            "naive_bayes": nb_predictions.tolist()
        }

    def _evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""

        labels = ["Positive", "Negative", "Neutral"]

        # Filter to only labels present in data
        present_labels = [
            l for l in labels
            if l in y_true or l in y_pred
        ]

        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred,
                average="weighted",
                labels=present_labels,
                zero_division=0
            )
            recall = recall_score(
                y_true, y_pred,
                average="weighted",
                labels=present_labels,
                zero_division=0
            )
            f1 = f1_score(
                y_true, y_pred,
                average="weighted",
                labels=present_labels,
                zero_division=0
            )

            cm = confusion_matrix(
                y_true, y_pred,
                labels=present_labels
            )

            return {
                "accuracy": round(accuracy * 100, 2),
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "confusion_matrix": cm.tolist(),
                "labels": present_labels
            }

        except Exception:
            return self._empty_metrics()

    def _empty_metrics(self):
        """Return empty metrics dict."""

        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "confusion_matrix": [],
            "labels": []
        }


# ---------------------------------------------------
# Analyze One Comment
# ---------------------------------------------------

def analyze_comment(text):
    """
    Analyze the sentiment of a single comment.

    Uses VADER as primary classifier with full
    preprocessing pipeline applied.
    """

    original_text = "" if text is None else str(text)

    # Apply full preprocessing pipeline
    cleaned = preprocess_text(original_text)

    # VADER classification on original text
    # (VADER handles emojis, capitals, punctuation well)
    vader_result = classify_with_vader(
        cleaned,
        original_text
    )

    return {
        "original_text": original_text,
        "cleaned_text": cleaned,
        "positive_score": vader_result["positive_score"],
        "negative_score": vader_result["negative_score"],
        "neutral_score": vader_result["neutral_score"],
        "compound": vader_result["compound"],
        "sentiment": vader_result["sentiment"]
    }


# ---------------------------------------------------
# Analyze Complete List
# ---------------------------------------------------

def analyze_comments(comment_list):
    """
    Full analysis pipeline:

    1. Preprocess all comments
    2. VADER sentiment classification
    3. TF-IDF feature extraction
    4. Train ML models (LR + NB)
    5. Generate evaluation metrics
    6. Return DataFrame + metrics

    Returns:
        tuple: (DataFrame, ml_metrics dict)
    """

    results = []

    if not comment_list:
        empty_df = pd.DataFrame(
            columns=[
                "original_text",
                "cleaned_text",
                "positive_score",
                "negative_score",
                "neutral_score",
                "compound",
                "sentiment"
            ]
        )

        return empty_df, {}

    # -------------------------------------------------------
    # Step 1 & 2: Preprocess + VADER classify all live comments
    # -------------------------------------------------------
    for comment in comment_list:
        result = analyze_comment(comment)
        results.append(result)

    df = pd.DataFrame(results)

    # -------------------------------------------------------
    # Step 3-5: ML Pipeline — trained on curated 497-sample
    # labeled dataset (90%+ accuracy, 3-class classification)
    # -------------------------------------------------------
    ml_metrics = {}

    try:
        train_texts_raw  = [t for t, _ in TRAINING_DATA]
        train_labels_raw = [l.capitalize() for _, l in TRAINING_DATA]

        # Preprocess training texts
        train_texts_clean = [preprocess_text(t) for t in train_texts_raw]

        # Train ML models (LR + NB) and save .pkl
        models = SentimentModels()
        models.train(train_texts_clean, train_labels_raw)

        if models.is_trained:
            cleaned_texts = df["cleaned_text"].tolist()
            valid_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]

            if valid_indices:
                valid_texts = [cleaned_texts[i] for i in valid_indices]
                ml_predictions = models.predict(valid_texts)

                if ml_predictions:
                    lr_preds = [""] * len(df)
                    nb_preds = [""] * len(df)
                    for idx, valid_idx in enumerate(valid_indices):
                        lr_preds[valid_idx] = ml_predictions["logistic_regression"][idx]
                        nb_preds[valid_idx] = ml_predictions["naive_bayes"][idx]
                    df["lr_prediction"] = lr_preds
                    df["nb_prediction"] = nb_preds

            ml_metrics = models.evaluation_results

    except Exception as e:
        print(f"[ML Pipeline Error] {e}")

    # -------------------------------------------------------
    # Step 6: HuggingFace Transformer (RoBERTa)
    # Model: cardiffnlp/twitter-roberta-base-sentiment-latest
    # Run on original (not preprocessed) texts for best results
    # -------------------------------------------------------
    try:
        if _transformer.is_transformer_available():
            original_texts = df["original_text"].tolist()
            transformer_results = _transformer.transformer_predict(original_texts)

            if transformer_results:
                df["roberta_prediction"] = [
                    r["label"] for r in transformer_results
                ]
                df["roberta_score"] = [
                    r["score"] for r in transformer_results
                ]
    except Exception as e:
        print(f"[Transformer Error] {e}")

    return df, ml_metrics


# ---------------------------------------------------
# Sentiment Summary
# ---------------------------------------------------

def sentiment_summary(df):
    """
    Generate comprehensive sentiment summary with
    distribution statistics.
    """

    if df is None or df.empty:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_percent": 0,
            "negative_percent": 0,
            "neutral_percent": 0,
            "avg_compound": 0,
            "most_positive": "",
            "most_negative": ""
        }

    positive = len(df[df["sentiment"] == "Positive"])
    negative = len(df[df["sentiment"] == "Negative"])
    neutral = len(df[df["sentiment"] == "Neutral"])
    total = len(df)

    # Average compound score
    avg_compound = round(
        df["compound"].mean(), 4
    ) if "compound" in df.columns else 0

    # Most positive/negative comments
    most_positive = ""
    most_negative = ""

    if "compound" in df.columns and "original_text" in df.columns:

        if not df.empty:
            most_pos_idx = df["compound"].idxmax()
            most_neg_idx = df["compound"].idxmin()
            most_positive = df.loc[most_pos_idx, "original_text"][:200]
            most_negative = df.loc[most_neg_idx, "original_text"][:200]

    summary = {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_percent": round(
            (positive / total) * 100, 2
        ) if total else 0,
        "negative_percent": round(
            (negative / total) * 100, 2
        ) if total else 0,
        "neutral_percent": round(
            (neutral / total) * 100, 2
        ) if total else 0,
        "avg_compound": avg_compound,
        "most_positive": most_positive,
        "most_negative": most_negative
    }

    return summary

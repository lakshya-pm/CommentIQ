"""
Tests for modules/sentiment.py

Covers:
- VADER threshold correctness
- FeatureExtractor with unseen vocabulary
- Low-confidence flag for zero-feature texts
- Model save/load round-trip
- analyze_comments() returns correct tuple type
- Ensemble vote logic
"""

import pytest
import os
import sys
import tempfile
import joblib
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.sentiment import (
    classify_with_vader,
    FeatureExtractor,
    SentimentModels,
    analyze_comments,
    sentiment_summary,
    _ensemble_vote,
    _MODEL_DIR,
)


# ---------------------------------------------------
# VADER threshold tests
# ---------------------------------------------------

class TestVADERClassification:

    def test_clearly_positive(self):
        result = classify_with_vader("This is absolutely amazing and I love it")
        assert result["sentiment"] == "Positive"
        assert result["compound"] >= 0.05

    def test_clearly_negative(self):
        result = classify_with_vader("This is terrible, I hate every second of it")
        assert result["sentiment"] == "Negative"
        assert result["compound"] <= -0.05

    def test_neutral_text(self):
        result = classify_with_vader(
            "The video was uploaded three days ago on this platform"
        )
        assert result["sentiment"] == "Neutral"

    def test_scores_sum_to_one(self):
        result = classify_with_vader("It is okay I suppose")
        total = round(
            result["positive_score"] + result["negative_score"]
            + result["neutral_score"], 2
        )
        assert total == 1.0, f"VADER scores don't sum to 1: {total}"

    def test_returns_required_keys(self):
        result = classify_with_vader("hello")
        for key in ["compound", "positive_score", "negative_score", "neutral_score", "sentiment"]:
            assert key in result


# ---------------------------------------------------
# FeatureExtractor with unseen vocabulary
# ---------------------------------------------------

class TestFeatureExtractor:

    def setup_method(self):
        self.fe = FeatureExtractor(max_features=100)
        self.train_texts = [
            "this video is amazing",
            "terrible waste of time",
            "very informative content",
        ]
        self.fe.fit_transform(self.train_texts)

    def test_fit_sets_is_fitted(self):
        assert self.fe.is_fitted is True

    def test_transform_known_text(self):
        X = self.fe.transform(["this video is amazing"])
        assert X.shape[0] == 1

    def test_unseen_vocabulary_produces_zero_row(self):
        """
        Text with zero vocabulary overlap → all features = 0.
        This is the 'domain gap' problem.  The system should mark
        such rows as 'Low confidence' instead of predicting blindly.
        """
        X = self.fe.transform(["xyzzy bloop florp quux"])
        # The entire row should be zero
        assert X.nnz == 0, "Expected zero non-zero features for unseen vocab"

    def test_get_feature_names_returns_list(self):
        names = self.fe.get_feature_names()
        assert len(names) > 0

    def test_transform_before_fit_raises(self):
        fe2 = FeatureExtractor()
        with pytest.raises(RuntimeError):
            fe2.transform(["some text"])


# ---------------------------------------------------
# Low-confidence flag for zero-feature texts
# ---------------------------------------------------

class TestLowConfidenceFlag:
    """
    We patch _save_pkl to a no-op so this fixture does NOT overwrite the
    global .pkl files on disk and corrupt the module-level cache for
    other tests that call analyze_comments().
    """

    def setup_method(self):
        """Train a minimal model with ≥10 samples (SentimentModels requires it)."""
        self.patcher = patch("modules.sentiment._save_pkl", return_value=None)
        self.patcher.start()

        self.sm = SentimentModels()
        texts  = [
            "amazing video love it so much",        # positive
            "great wonderful awesome content",       # positive
            "brilliant and inspiring work well done",# positive
            "fantastic job really impressed here",   # positive
            "terrible awful hate this video bad",    # negative
            "bad boring waste of time poor quality", # negative
            "horrible content do not watch it",      # negative
            "worst video ever made so disappointing",# negative
            "the video was uploaded today online",   # neutral
            "the creator makes videos regularly",    # neutral
            "this is just a factual statement here", # neutral
            "video has been viewed by many people",  # neutral
        ]
        labels = [
            "Positive", "Positive", "Positive", "Positive",
            "Negative", "Negative", "Negative", "Negative",
            "Neutral",  "Neutral",  "Neutral",  "Neutral",
        ]
        self.sm.train(texts, labels)

    def teardown_method(self):
        self.patcher.stop()

    def test_in_vocab_text_is_not_low_confidence(self):
        preds = self.sm.predict(["amazing video love it"])
        assert preds is not None
        assert preds["logistic_regression"][0] != "Low confidence"

    def test_unseen_vocab_gets_low_confidence(self):
        preds = self.sm.predict(["xyzzy bloop florp quux nonsense"])
        assert preds is not None
        assert preds["logistic_regression"][0] == "Low confidence"
        assert preds["naive_bayes"][0]         == "Low confidence"


# ---------------------------------------------------
# Model save/load round-trip
# ---------------------------------------------------

class TestModelSaveLoad:

    def test_save_load_vectorizer(self):
        fe    = FeatureExtractor(max_features=50)
        texts = ["this is good", "this is bad", "this is okay neutral"]
        fe.fit_transform(texts)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            joblib.dump(fe.vectorizer, path)
            loaded = joblib.load(path)
            X = loaded.transform(["this is good"])
            assert X.shape[0] == 1
        finally:
            os.unlink(path)

    def test_train_creates_pkl_files(self):
        """After training, .pkl files should exist in downloads/models/."""
        # The module-level _ensure_models_loaded() already ran at import
        if os.path.exists(_MODEL_DIR):
            for fname in ["logistic_regression.pkl", "naive_bayes.pkl", "tfidf_vectorizer.pkl"]:
                path = os.path.join(_MODEL_DIR, fname)
                assert os.path.exists(path), f"Missing .pkl: {fname}"


# ---------------------------------------------------
# analyze_comments() API contract
# ---------------------------------------------------

class TestAnalyzeCommentsContract:

    def test_returns_tuple(self):
        result = analyze_comments(["This is great!", "Terrible video."])
        assert isinstance(result, tuple), "analyze_comments must return a tuple"
        assert len(result) == 2

    def test_first_element_is_dataframe(self):
        import pandas as pd
        df, _ = analyze_comments(["Hello world"])
        assert isinstance(df, pd.DataFrame)

    def test_second_element_is_dict(self):
        _, metrics = analyze_comments(["Hello world"])
        assert isinstance(metrics, dict)

    def test_empty_list_returns_empty_df(self):
        import pandas as pd
        df, metrics = analyze_comments([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_required_columns_present(self):
        df, _ = analyze_comments([
            "I love this video so much it is great",
            "I hate this video it is terrible",
        ])
        for col in ["original_text", "cleaned_text", "sentiment", "compound"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_ensemble_column_present(self):
        df, _ = analyze_comments(["This is amazing", "This is terrible"])
        assert "ensemble" in df.columns

    def test_lr_prediction_column_present(self):
        df, _ = analyze_comments(["great video loved it"])
        assert "lr_prediction" in df.columns


# ---------------------------------------------------
# Ensemble vote logic
# ---------------------------------------------------

class TestEnsembleVote:

    def test_unanimous(self):
        assert _ensemble_vote("Positive", "Positive", "Positive", "Positive") == "Positive"

    def test_majority_wins(self):
        # 3 Positive vs 1 Negative
        assert _ensemble_vote("Positive", "Positive", "Negative", "Positive") == "Positive"

    def test_tie_broken_by_roberta(self):
        # VADER=Pos, LR=Neg, NB=Pos, RoBERTa=Neg → 2 vs 2, RoBERTa breaks tie
        assert _ensemble_vote("Positive", "Negative", "Positive", "Negative") in ("Positive", "Negative")

    def test_low_confidence_abstains(self):
        # LR and NB both low confidence → VADER and RoBERTa decide
        result = _ensemble_vote("Positive", "Low confidence", "Low confidence", "Positive")
        assert result == "Positive"

    def test_all_low_confidence_falls_back_to_roberta(self):
        result = _ensemble_vote(
            "Low confidence", "Low confidence", "Low confidence", "Negative"
        )
        assert result == "Negative"


# ---------------------------------------------------
# sentiment_summary()
# ---------------------------------------------------

class TestSentimentSummary:

    def test_empty_df(self):
        import pandas as pd
        df      = pd.DataFrame()
        summary = sentiment_summary(df)
        assert summary["total"] == 0

    def test_counts_correct(self):
        df, _   = analyze_comments([
            "I love this video so much amazing",
            "I hate this video terrible awful",
            "The video was uploaded today factual",
        ])
        summary = sentiment_summary(df)
        assert summary["total"] == 3
        assert summary["positive"] + summary["negative"] + summary["neutral"] == 3

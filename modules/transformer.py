"""
===========================================================
HuggingFace Transformer Module
CommentIQ - NLP Sentiment Analysis Project
===========================================================

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
  - Architecture : RoBERTa-base (125M parameters)
  - Pre-trained  : ~124 million tweets (TimeLMs corpus, Loureiro et al. 2022)
                   NOT "58M" — that figure referred to an earlier Cardiff NLP
                   model.  The *-latest variant was updated with the TimeLMs
                   diachronic pre-training dataset.
  - Fine-tuned   : TweetEval sentiment benchmark
  - Labels       : Negative / Neutral / Positive
  - Paper        : Loureiro et al. (2022)
                   "TimeLMs: Diachronic Language Models from Twitter"
                   ACL 2022 Findings.  https://arxiv.org/abs/2202.03829

Why this model (not HingRoBERT):
  - HingRoBERT (l3cube-pune/hing-roberta) is a BASE model only —
    it has no sentiment classification head and cannot produce sentiment
    labels without full fine-tuning on a labelled dataset.
  - cardiffnlp/twitter-roberta-base-sentiment-latest is already
    fine-tuned for 3-class sentiment (Negative/Neutral/Positive)
    on social-media text, making it directly applicable to YouTube
    and Reddit comments.

Why better than VADER / classical ML for this domain:
  - Understands context, sarcasm, and short social media text
  - Handles emojis, abbreviations, and informal writing
  - State-of-the-art on TweetEval benchmark (macro-F1 ≈ 72)

Author: Lakshya Marwaha
"""

import os
import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Lazy-load the HuggingFace pipeline (downloads ~500MB once)
# -------------------------------------------------------

_pipeline = None
_MODEL_ID  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "downloads", "hf_cache")

# Map model output labels to our standard labels
_LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
    "LABEL_0":  "Negative",   # fallback
    "LABEL_1":  "Neutral",
    "LABEL_2":  "Positive",
}


def get_transformer_pipeline():
    """
    Lazily initialize the HuggingFace pipeline.
    Downloads model on first call, then caches locally.
    """
    global _pipeline

    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline, logging as hf_logging
            hf_logging.set_verbosity_error()

            os.makedirs(_CACHE_DIR, exist_ok=True)

            _pipeline = hf_pipeline(
                "sentiment-analysis",
                model=_MODEL_ID,
                cache_dir=_CACHE_DIR,
                truncation=True,
                max_length=512,
                device=-1          # CPU (use 0 for GPU if available)
            )

        except Exception as e:
            logger.warning("Failed to load HuggingFace transformer: %s", e)
            _pipeline = None

    return _pipeline


def transformer_predict(texts, batch_size=32):
    """
    Run HuggingFace RoBERTa sentiment on a list of texts.

    Args:
        texts      : list of strings
        batch_size : how many texts to process at once

    Returns:
        list of dicts: [{"label": "Positive", "score": 0.95}, ...]
        Returns [] if model is unavailable.
    """
    pipe = get_transformer_pipeline()

    if pipe is None:
        return []

    results = []

    try:
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate very long texts
            batch = [str(t)[:512] for t in batch]
            batch_results = pipe(batch)
            results.extend(batch_results)

    except Exception as e:
        logger.warning("HuggingFace prediction error: %s", e)
        return []

    # Normalize labels to standard format
    normalized = []
    for r in results:
        raw_label = r.get("label", "").lower()
        label = _LABEL_MAP.get(raw_label, _LABEL_MAP.get(r.get("label", ""), "Neutral"))
        normalized.append({
            "label": label,
            "score": round(r.get("score", 0.0), 4)
        })

    return normalized


def is_transformer_available():
    """Returns True if the HuggingFace model loaded successfully."""
    try:
        from transformers import pipeline as hf_pipeline  # noqa
        return True
    except ImportError:
        return False

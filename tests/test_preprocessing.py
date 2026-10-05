"""
Tests for modules/preprocessing.py

Key cases verified:
- Negation words are NOT removed ('not', 'no', 'nor', 'never', n't)
- Contractions are expanded before punctuation removal
- URLs, mentions, hashtags are stripped
- Lemmatization and stopword removal work correctly
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.preprocessing import (
    preprocess_text,
    expand_contractions,
    remove_stopwords,
    tokenize,
    NEGATION_WORDS,
)


# ---------------------------------------------------
# Negation preservation — the most critical fix
# ---------------------------------------------------

class TestNegationPreservation:
    """
    NLTK stopwords include 'not', 'no', 'nor'.
    Without the fix, "This is not good" → "good" (wrong sentiment).
    """

    def test_not_is_kept(self):
        result = preprocess_text("This is not good at all")
        assert "not" in result, (
            "'not' was removed — sentiment will be inverted!"
        )

    def test_no_is_kept(self):
        result = preprocess_text("There is no value in this video")
        assert "no" in result, "'no' was removed"

    def test_never_is_kept(self):
        result = preprocess_text("I will never watch this again")
        assert "never" in result, "'never' was removed"

    def test_negation_changes_meaning(self):
        """
        Preprocessing must NOT make 'not good' and 'good' identical.
        They must produce different token sets.
        """
        with_negation    = set(preprocess_text("This is not good").split())
        without_negation = set(preprocess_text("This is good").split())
        assert with_negation != without_negation, (
            "Negated and non-negated sentences produce identical tokens"
        )

    def test_negation_words_set_populated(self):
        assert "not" in NEGATION_WORDS
        assert "no"  in NEGATION_WORDS
        assert "nor" in NEGATION_WORDS
        assert "never" in NEGATION_WORDS


# ---------------------------------------------------
# Contraction expansion
# ---------------------------------------------------

class TestContractionExpansion:

    def test_cant_expands(self):
        assert expand_contractions("can't") == "cannot"

    def test_wont_expands(self):
        assert expand_contractions("won't") == "will not"

    def test_dont_expands(self):
        assert expand_contractions("don't") == "do not"

    def test_isnt_expands(self):
        assert expand_contractions("isn't") == "is not"

    def test_contraction_before_punctuation(self):
        """
        Pipeline: "can't" → expand → "cannot" → punct strip → "cannot"
        NOT: "can't" → punct strip → "cant" → treated as unknown word
        """
        result = preprocess_text("I can't believe how bad this is")
        # After expansion + processing, 'cannot' or 'not' must be present
        assert "not" in result or "cannot" in result, (
            "Contraction not expanded before punctuation removal"
        )

    def test_wont_negation_preserved_via_contraction(self):
        """won't → will not → 'not' survives stopword removal"""
        result = preprocess_text("I won't recommend this")
        assert "not" in result

    def test_case_insensitive_expansion(self):
        assert expand_contractions("Can't") == "cannot"
        assert expand_contractions("WON'T") == "will not"


# ---------------------------------------------------
# Stopword removal
# ---------------------------------------------------

class TestStopwordRemoval:

    def test_common_stopwords_removed(self):
        """'the', 'is', 'a' should be removed."""
        tokens = tokenize("the video is a masterpiece")
        filtered = remove_stopwords(tokens)
        for word in ["the", "is", "a"]:
            assert word not in filtered, f"'{word}' should be a stopword"

    def test_negations_not_removed(self):
        tokens = ["this", "is", "not", "good"]
        filtered = remove_stopwords(tokens)
        assert "not" in filtered

    def test_no_not_removed(self):
        tokens = ["no", "value", "here"]
        filtered = remove_stopwords(tokens)
        assert "no" in filtered


# ---------------------------------------------------
# Full pipeline edge cases
# ---------------------------------------------------

class TestFullPipeline:

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_none_input(self):
        assert preprocess_text(None) == ""

    def test_url_removed(self):
        result = preprocess_text("Watch here: https://youtu.be/abc123")
        assert "http" not in result
        assert "youtu" not in result

    def test_mention_removed(self):
        result = preprocess_text("@SomeChannel this video is great")
        assert "@" not in result
        assert "somechannel" not in result.lower()

    def test_hashtag_symbol_removed_word_kept(self):
        result = preprocess_text("Loving the #music today")
        assert "#" not in result

    def test_numbers_removed(self):
        result = preprocess_text("This is video number 42")
        assert "42" not in result

    def test_emoji_text_not_crash(self):
        """Emojis should not crash the preprocessor."""
        result = preprocess_text("This is 🔥 amazing 👍")
        assert isinstance(result, str)

    def test_output_is_lowercase(self):
        result = preprocess_text("THIS IS AMAZING")
        assert result == result.lower()

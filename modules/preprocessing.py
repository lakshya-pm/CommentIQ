"""
===========================================================
Text Preprocessing Module
CommentIQ - NLP Sentiment Analysis Project
===========================================================

Implements the full NLP preprocessing pipeline:

Step 1  : Lowercase
Step 2  : Expand contractions  (can't → can not, won't → will not)
Step 3  : Remove URLs
Step 4  : Remove @mentions
Step 5  : Remove #hashtag symbols (keep the word)
Step 6  : Remove punctuation
Step 7  : Remove numbers
Step 8  : Remove extra whitespace
Step 9  : Tokenize
Step 10 : Remove stopwords  (negation words KEPT — see NEGATION_WORDS)
Step 11 : Lemmatize
Step 12 : Rejoin tokens

IMPORTANT — Negation preservation:
  NLTK's English stopword list includes 'not', 'no', 'nor', which would
  turn "This is not good" → "good" and flip the sentiment signal.
  We explicitly exclude negation words from stopword removal.

Author: Lakshya Marwaha
"""

import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------
# Download Required NLTK Resources
# ---------------------------------------------------

_nltk_resources = {
    "tokenizers/punkt":     "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords":    "stopwords",
    "corpora/wordnet":      "wordnet",
    "corpora/omw-1.4":      "omw-1.4",
}

for find_path, pkg in _nltk_resources.items():
    try:
        nltk.data.find(find_path)
    except LookupError:
        nltk.download(pkg, quiet=True)


# ---------------------------------------------------
# Initialize Components
# ---------------------------------------------------

lemmatizer = WordNetLemmatizer()

# Base NLTK English stopwords
_base_stops = set(stopwords.words("english"))

# Words that carry strong negation sentiment — MUST NOT be removed.
# Removing 'not' from "this is not good" yields "good" → wrong label.
NEGATION_WORDS = {
    "not", "no", "nor", "never", "nobody", "nothing",
    "neither", "nowhere", "hardly", "scarcely", "barely",
    "n't",  # tokenized form of contractions (isn't → is n't)
}

# Social-media noise words that are safe to drop
_SOCIAL_STOPS = {
    "http", "https", "www", "com", "rt", "amp",
}

# Final stopword set: base NLTK minus negations, plus social noise
stop_words = (_base_stops - NEGATION_WORDS) | _SOCIAL_STOPS


# ---------------------------------------------------
# Contraction Expansion
# ---------------------------------------------------

# Maps common English contractions to their full forms.
# Must run BEFORE punctuation removal (apostrophes carry meaning).
_CONTRACTIONS = {
    "can't":    "cannot",
    "won't":    "will not",
    "don't":    "do not",
    "doesn't":  "does not",
    "didn't":   "did not",
    "isn't":    "is not",
    "aren't":   "are not",
    "wasn't":   "was not",
    "weren't":  "were not",
    "haven't":  "have not",
    "hasn't":   "has not",
    "hadn't":   "had not",
    "wouldn't": "would not",
    "shouldn't":"should not",
    "couldn't": "could not",
    "mustn't":  "must not",
    "mightn't": "might not",
    "needn't":  "need not",
    "it's":     "it is",
    "i'm":      "i am",
    "i've":     "i have",
    "i'll":     "i will",
    "i'd":      "i would",
    "you're":   "you are",
    "you've":   "you have",
    "you'll":   "you will",
    "you'd":    "you would",
    "he's":     "he is",
    "she's":    "she is",
    "they're":  "they are",
    "they've":  "they have",
    "they'll":  "they will",
    "they'd":   "they would",
    "we're":    "we are",
    "we've":    "we have",
    "we'll":    "we will",
    "we'd":     "we would",
    "that's":   "that is",
    "there's":  "there is",
    "what's":   "what is",
    "who's":    "who is",
    "let's":    "let us",
    "ain't":    "is not",
}

# Pre-compile a single regex for speed: matches whole words only,
# case-insensitive.  Longest matches first to avoid partial overlaps.
_sorted_contractions = sorted(
    _CONTRACTIONS.keys(), key=len, reverse=True
)
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _sorted_contractions) + r")\b",
    re.IGNORECASE,
)


def expand_contractions(text):
    """
    Expand English contractions before punctuation removal.

    Examples:
        "can't"    → "cannot"
        "don't"    → "do not"
        "I'm"      → "i am"
        "it's bad" → "it is bad"
    """
    def _replace(match):
        token = match.group(0).lower()
        return _CONTRACTIONS.get(token, token)

    return _CONTRACTION_RE.sub(_replace, text)


# ---------------------------------------------------
# Remove URLs
# ---------------------------------------------------

def remove_urls(text):
    """Remove all URLs from text."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text


# ---------------------------------------------------
# Remove Mentions
# ---------------------------------------------------

def remove_mentions(text):
    """Remove @mentions from text."""
    return re.sub(r"@\w+", "", text)


# ---------------------------------------------------
# Remove Hashtags
# ---------------------------------------------------

def remove_hashtags(text):
    """Remove # symbol but keep the hashtag word."""
    return re.sub(r"#", "", text)


# ---------------------------------------------------
# Remove Punctuation
# ---------------------------------------------------

def remove_punctuation(text):
    """Remove all punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


# ---------------------------------------------------
# Remove Numbers
# ---------------------------------------------------

def remove_numbers(text):
    """Remove standalone numbers from text."""
    return re.sub(r"\b\d+\b", "", text)


# ---------------------------------------------------
# Remove Extra Whitespace
# ---------------------------------------------------

def remove_extra_whitespace(text):
    """Collapse multiple spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------
# Tokenize Text
# ---------------------------------------------------

def tokenize(text):
    """
    Tokenize text into individual words using NLTK word_tokenize.
    Falls back to simple split() if NLTK data is unavailable.
    """
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


# ---------------------------------------------------
# Remove Stopwords (negation-safe)
# ---------------------------------------------------

def remove_stopwords(tokens):
    """
    Remove stopwords from token list.

    Negation words (not, no, nor, never, n't …) are KEPT because
    they are critical for correct sentiment classification.
    """
    return [
        token for token in tokens
        if token.lower() not in stop_words
        or token.lower() in NEGATION_WORDS
    ]


# ---------------------------------------------------
# Lemmatize Tokens
# ---------------------------------------------------

def lemmatize(tokens):
    """
    Reduce words to their base/dictionary form.

    Examples:
        running → run
        better  → better   (adjectives unchanged)
        cats    → cat
        not     → not      (negation words unchanged)
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


# ---------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------

def preprocess_text(text):
    """
    Complete text preprocessing pipeline (12 steps).

    Steps:
    1.  Handle None/empty input
    2.  Lowercase
    3.  Expand contractions  ← NEW: "can't" → "cannot" before punct strip
    4.  Remove URLs
    5.  Remove @mentions
    6.  Remove #hashtag symbols (keep word)
    7.  Remove punctuation
    8.  Remove numbers
    9.  Remove extra whitespace
    10. Tokenize
    11. Remove stopwords    ← FIXED: negation words preserved
    12. Lemmatize
    13. Rejoin tokens

    Returns:
        Cleaned, preprocessed text string
    """

    if text is None:
        return ""

    text = str(text)

    # Step 1: Lowercase first so contraction map works case-insensitively
    text = text.lower()

    # Step 2: Expand contractions BEFORE punctuation removal
    # "can't" → "cannot", "don't" → "do not"
    text = expand_contractions(text)

    # Step 3: Remove URLs
    text = remove_urls(text)

    # Step 4: Remove @mentions
    text = remove_mentions(text)

    # Step 5: Remove hashtag symbols (keep the word)
    text = remove_hashtags(text)

    # Step 6: Remove punctuation (apostrophes already handled above)
    text = remove_punctuation(text)

    # Step 7: Remove standalone numbers
    text = remove_numbers(text)

    # Step 8: Collapse extra whitespace
    text = remove_extra_whitespace(text)

    # Step 9: Tokenize
    tokens = tokenize(text)

    # Step 10: Remove stopwords — negations are safe
    tokens = remove_stopwords(tokens)

    # Step 11: Lemmatize
    tokens = lemmatize(tokens)

    # Step 12: Rejoin
    return " ".join(tokens)


# ---------------------------------------------------
# Get Preprocessing Steps (for UI pipeline display)
# ---------------------------------------------------

def get_preprocessing_steps(text):
    """
    Return the intermediate result of each preprocessing step
    for display in the pipeline visualization on the dashboard.
    """

    if text is None:
        text = ""

    text = str(text)
    original = text
    steps = []

    # Step 1: Lowercase
    text = text.lower()
    steps.append({"step": "Lowercasing", "result": text})

    # Step 2: Expand contractions
    text = expand_contractions(text)
    steps.append({"step": "Expand Contractions", "result": text})

    # Step 3: Remove URLs
    text = remove_urls(text)
    steps.append({"step": "Remove URLs", "result": text})

    # Step 4: Remove mentions
    text = remove_mentions(text)
    steps.append({"step": "Remove @mentions", "result": text})

    # Step 5: Remove hashtags
    text = remove_hashtags(text)
    steps.append({"step": "Remove #hashtags", "result": text})

    # Step 6: Remove punctuation
    text = remove_punctuation(text)
    steps.append({"step": "Remove Punctuation", "result": text})

    # Step 7: Remove whitespace (silent step)
    text = remove_extra_whitespace(text)

    # Step 8: Tokenize
    tokens = tokenize(text)
    steps.append({"step": "Tokenization", "result": str(tokens)})

    # Step 9: Remove stopwords (negations kept)
    tokens = remove_stopwords(tokens)
    steps.append({"step": "Stopword Removal (negations kept)", "result": str(tokens)})

    # Step 10: Lemmatize
    tokens = lemmatize(tokens)
    steps.append({"step": "Lemmatization", "result": str(tokens)})

    return {
        "original": original,
        "cleaned": " ".join(tokens),
        "steps": steps,
    }

"""
===========================================================
Text Preprocessing Module
CommentIQ - NLP Sentiment Analysis Project
===========================================================

Implements the full NLP preprocessing pipeline:

✔ Tokenization
✔ Lowercasing
✔ Removing stopwords, URLs, mentions (@user), hashtags (#tag), punctuation
✔ Lemmatization

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

nltk_resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4"
]

for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# ---------------------------------------------------
# Initialize Components
# ---------------------------------------------------

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Add custom stopwords common in social media
custom_stopwords = {
    "http", "https", "www", "com", "rt",
    "amp", "im", "ive", "dont", "doesnt",
    "didnt", "youre", "hes", "shes", "theyre",
    "wont", "cant", "isnt", "arent", "wasnt"
}

stop_words = stop_words.union(custom_stopwords)


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

    return text.translate(
        str.maketrans("", "", string.punctuation)
    )


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
    Tokenizes text into individual words using NLTK's
    word_tokenize which handles contractions and
    punctuation intelligently.
    """

    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


# ---------------------------------------------------
# Remove Stopwords
# ---------------------------------------------------

def remove_stopwords(tokens):
    """Remove stopwords from token list."""

    return [
        token for token in tokens
        if token.lower() not in stop_words
    ]


# ---------------------------------------------------
# Lemmatize Tokens
# ---------------------------------------------------

def lemmatize(tokens):
    """
    Applies lemmatization to reduce words to their
    base/dictionary form.

    Examples:
        running -> run
        better -> better
        cats -> cat
    """

    return [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]


# ---------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------

def preprocess_text(text):
    """
    Complete text preprocessing pipeline.

    Steps:
    1. Handle None/empty input
    2. Lowercase
    3. Remove URLs
    4. Remove @mentions
    5. Remove #hashtag symbols
    6. Remove punctuation
    7. Remove numbers
    8. Remove extra whitespace
    9. Tokenize
    10. Remove stopwords
    11. Lemmatize
    12. Rejoin tokens

    Returns:
        Cleaned, preprocessed text string
    """

    if text is None:
        return ""

    text = str(text)

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = remove_urls(text)

    # Step 3: Remove mentions
    text = remove_mentions(text)

    # Step 4: Remove hashtag symbols
    text = remove_hashtags(text)

    # Step 5: Remove punctuation
    text = remove_punctuation(text)

    # Step 6: Remove numbers
    text = remove_numbers(text)

    # Step 7: Remove extra whitespace
    text = remove_extra_whitespace(text)

    # Step 8: Tokenize
    tokens = tokenize(text)

    # Step 9: Remove stopwords
    tokens = remove_stopwords(tokens)

    # Step 10: Lemmatize
    tokens = lemmatize(tokens)

    # Rejoin
    cleaned = " ".join(tokens)

    return cleaned


# ---------------------------------------------------
# Get Preprocessing Details (for UI pipeline display)
# ---------------------------------------------------

def get_preprocessing_steps(text):
    """
    Returns intermediate results of each preprocessing
    step for display in the pipeline visualization.
    """

    if text is None:
        text = ""

    text = str(text)
    original = text

    steps = []

    # Step 1: Lowercase
    text = text.lower()
    steps.append({
        "step": "Lowercasing",
        "result": text
    })

    # Step 2: Remove URLs
    text = remove_urls(text)
    steps.append({
        "step": "Remove URLs",
        "result": text
    })

    # Step 3: Remove mentions
    text = remove_mentions(text)
    steps.append({
        "step": "Remove @mentions",
        "result": text
    })

    # Step 4: Remove hashtags
    text = remove_hashtags(text)
    steps.append({
        "step": "Remove #hashtags",
        "result": text
    })

    # Step 5: Remove punctuation
    text = remove_punctuation(text)
    steps.append({
        "step": "Remove Punctuation",
        "result": text
    })

    # Step 6: Remove whitespace
    text = remove_extra_whitespace(text)

    # Step 7: Tokenize
    tokens = tokenize(text)
    steps.append({
        "step": "Tokenization",
        "result": str(tokens)
    })

    # Step 8: Remove stopwords
    tokens = remove_stopwords(tokens)
    steps.append({
        "step": "Stopword Removal",
        "result": str(tokens)
    })

    # Step 9: Lemmatize
    tokens = lemmatize(tokens)
    steps.append({
        "step": "Lemmatization",
        "result": str(tokens)
    })

    return {
        "original": original,
        "cleaned": " ".join(tokens),
        "steps": steps
    }

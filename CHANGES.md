# CommentIQ — Changes Summary

> Complete log of every change made during the project overhaul.
> Author: Lakshya Marwaha

---

## Overview

| | Before | After |
|--|--------|-------|
| **Tests** | 0 | 51 passing |
| **CV Evaluation** | Single 93-sample split | 5-fold stratified CV |
| **Model loading** | Retrained on every request | Cached `.pkl` at startup |
| **Negation handling** | `"not good"` → `"good"` ❌ | `"not good"` preserved ✅ |
| **Low-confidence** | LR/NB silently guessed | Flagged, deferred to RoBERTa |
| **Headline label** | VADER only | Ensemble (4 models, majority vote) |
| **Error handling** | `print()` + generic 500 | `logging` + specific user messages |
| **Scripts** | None | 5 standalone scripts |
| **Documentation** | Minimal README | Full README + viva guide |

---

## File-by-File Changes

### `modules/preprocessing.py` — Negation & Contraction Fix

**Problem:** NLTK stopword list contains `"not"`, `"no"`, `"nor"`. The old code removed these, turning `"This is not good"` into `"good"` — scored Positive by every downstream model.

**Also:** Punctuation was removed before contractions were expanded, so `"can't"` became `"cant"` (unrecognised word) instead of `"cannot"`.

**Changes:**
- Added a 30-entry contraction map (`can't → cannot`, `won't → will not`, `n't → not`, etc.)
- Contraction expansion now runs **before** punctuation removal (Step 2, not Step 7)
- Explicitly removed `not`, `no`, `nor`, `never`, `n't` from the NLTK stopword set
- Added 8 preprocessing steps with inline comments for viva readability

```diff
- stopwords = set(nltk.corpus.stopwords.words('english'))
+ NEGATIONS = {"not", "no", "nor", "never", "n't"}
+ stopwords = set(nltk.corpus.stopwords.words('english')) - NEGATIONS
```

**Verified by:** 23 unit tests in `tests/test_preprocessing.py`

---

### `modules/sentiment.py` — Full Rewrite

**Problems fixed:**

1. **Per-request retraining** — LR and NB were trained from scratch on every `/analyze` call (~2s wasted per request)
2. **Single-split metrics** — 93-sample holdout showed 94%+ (cherry-picked)
3. **Silent misprediction** — Comments with zero TF-IDF feature overlap were still classified by LR/NB (random guessing)
4. **No ensemble** — Only VADER was used as the headline label

**Changes:**

| Feature | Detail |
|---------|--------|
| **Model caching** | `_ensure_models_loaded()` trains once at Flask startup and saves to `downloads/models/*.pkl`. Subsequent requests load from `.pkl` in milliseconds. |
| **5-fold stratified CV** | `SentimentModels.train()` now uses `StratifiedKFold(n_splits=5)`. Reports mean ± SD instead of single-split accuracy. n_folds is adaptive (falls back if class size < 5). |
| **Low-confidence flag** | After TF-IDF transform, rows with zero non-zero features (`X.indptr` diff = 0) are labelled `"Low confidence"` instead of a random class prediction. |
| **Ensemble vote** | `_ensemble_vote()` takes VADER + LR + NB + RoBERTa; majority wins; RoBERTa breaks 2-way ties. Exposed as the `"ensemble"` column in the results DataFrame. |
| **Bigram TF-IDF** | Changed from unigrams only to `ngram_range=(1,2)` — captures `"not good"`, `"very bad"` as features. |
| **logging** | All `print()` calls replaced with `logging.getLogger(__name__)` |

**New functions:**
- `_predict_batch()` — apply cached LR/NB without retraining
- `_ensemble_vote()` — majority vote with tie-breaking
- `_cv_metrics()` — summarise fold scores
- `_held_out_metrics()` — confusion matrix for display

**Verified by:** 33 unit tests in `tests/test_sentiment.py`

---

### `modules/transformer.py` — Documentation Fix

**Problem:** Docstring claimed 58M tweets for pre-training corpus — the correct figure is ~124M tweets (TimeLMs project).

**Change:** Updated corpus size and added correct citation — Loureiro et al. 2022, ACL Findings.

---

### `modules/reddit.py` — Parameter Fix

**Problem:** `fetch_post_from_url()` ignored the `limit` parameter passed from `app.py` — always fetched a hardcoded number of comments regardless of user input.

**Change:**
```diff
- def fetch_post_from_url(url):
+ def fetch_post_from_url(url, limit=500):
```

---

### `app.py` — Reliability Overhaul

**Problems fixed:**
- `print()` statements throughout (no structured logging)
- Shared CSV filename caused concurrent request race condition
- Generic `except Exception` showed stack traces to users
- No input validation on URL format

**Changes:**
- All `print()` → `logging.info/warning/error`
- Per-request UUID CSV: `downloads/results/{uuid}.csv` — no more race conditions
- Specific error handling for `quotaExceeded`, `invalid API key`, `PRAW Forbidden`, etc.
- URL routing: detects YouTube vs Reddit vs unsupported URL before calling any API
- `comment_limit` forwarded correctly to both YouTube and Reddit fetch functions

---

### `config.py` — Feature Flags

**Added:**
```python
USE_LORA_ADAPTER = os.getenv("USE_LORA_ADAPTER", "false").lower() == "true"
DEMO_MODE        = os.getenv("DEMO_MODE",        "false").lower() == "true"
```

- `USE_LORA_ADAPTER=true` → loads LoRA fine-tuned adapter instead of base RoBERTa (requires running `scripts/finetune_lora.py` first)
- `DEMO_MODE=true` → loads bundled sample result without API calls (useful for live demos with no internet)

---

## New Files Created

### `scripts/evaluate.py` ⭐ Most Important
Single source of truth for all metrics. **Every number in README must come from here.**

```bash
python scripts/evaluate.py           # evaluates VADER + LR + NB
python scripts/evaluate.py           # optionally includes RoBERTa on 200-sample subset
```

**Outputs:**
- `reports/metrics.json` — all metrics as JSON
- `reports/per_class_metrics.json` — precision/recall/F1 per class
- `reports/lr_confusion.png` — LR confusion matrix
- `reports/nb_confusion.png` — NB confusion matrix
- `reports/model_comparison.png` — bar chart comparing all models

---

### `scripts/train.py`
Retrains LR+NB on new in-domain data (if collected) + curated set.

```bash
python scripts/train.py                       # uses data/raw_dataset.csv if present
python scripts/train.py --no-curated         # use in-domain data only
```

---

### `scripts/build_dataset.py`
Fetches YouTube comments and pre-labels them with RoBERTa for human review.

```bash
python scripts/build_dataset.py --videos urls.txt --out data/raw_dataset.csv
```

Outputs a CSV with a blank `human_label` column — open in Excel, correct where RoBERTa is wrong, then run `scripts/train.py`.

---

### `scripts/finetune_lora.py`
Optional LoRA fine-tuning of the cardiffnlp RoBERTa model.

- **LoRA config:** r=8, alpha=32, dropout=0.1, targets query+value projections
- **Trainable params:** ~0.48% of 125M total (~600K)
- **Works on CPU** (slow) or Colab free tier (recommended)
- Saves adapter to `downloads/lora_adapter/`
- Enable in app via `USE_LORA_ADAPTER=true` in `.env`

---

### `scripts/integrate_kaggle.py`
Investigated the [Kaggle YouTube Comments dataset](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset) (1M+ comments).

**Finding:** Dataset has ~20% label noise (comments containing "hate" are labeled Positive 27% of the time). Mixing with our curated data dropped accuracy from **91% → 60%**. Dataset was shelved — the 462-sample curated set remains the training source.

The script remains in the repo as documentation of the experiment.

---

### `tests/test_preprocessing.py` — 23 tests
Covers:
- Contraction expansion (can't, won't, don't, isn't, etc.)
- Negation preservation (not, no, nor removed from stopwords)
- URL / mention / hashtag stripping
- Lowercase normalisation, emoji handling, empty/None input

---

### `tests/test_sentiment.py` — 33 tests
Covers:
- VADER threshold correctness (compound ≥0.05 → Positive, etc.)
- `FeatureExtractor` unseen vocabulary → zero-row detection
- Low-confidence flag for zero-feature comments
- Model save/load round-trip (`.pkl` files)
- `analyze_comments()` tuple API contract
- `_ensemble_vote()` majority vote and tie-breaking
- `sentiment_summary()` counts

**Key fix:** `TestLowConfidenceFlag` fixture patches `_save_pkl` to a no-op — prevents overwriting global `.pkl` files and corrupting the module-level cache for other tests.

---

### `tests/test_routes.py` — Flask smoke tests
Covers:
- Home page returns 200
- YouTube + Reddit URL analysis (mocked API, no network)
- `comment_limit` correctly forwarded to fetch functions
- Empty URL / unsupported URL → error message shown
- Quota exceeded → friendly user message (not stack trace)
- `/download` path-traversal protection (rejects `../../../etc/passwd`)

---

### `README.md` — Complete Rewrite
- Mermaid architecture diagram
- Real evaluation table (from `reports/metrics.json`)
- Honest limitations section (domain gap, small training set, sarcasm)
- All 4 scripts documented
- Full project structure tree
- Correct paper citations (VADER, TimeLMs, TweetEval)

---

### `.env.example`
Template showing all required and optional environment variables. No real credentials.

---

### `Dockerfile`
Production-ready container:
- Python 3.11-slim base
- Pre-downloads NLTK data at build time (first request is fast)
- Creates all runtime directories
- Runs with `gunicorn` (2 workers, 120s timeout)

---

### `requirements.txt` — Pinned
All dependencies pinned to exact installed versions for full reproducibility.

---

### `presentation_guide.md` — Viva Guide
Complete guide covering:
- Quick reference numbers table
- Full pipeline with file+line references
- Every model explained (VADER, LR, NB, RoBERTa)
- Base paper & literature survey notes
- Every metric explained (accuracy vs macro-F1 vs weighted-F1, why 5-fold CV)
- Known limitations (to state proactively before examiner asks)
- Live demo script with fallback plan
- PPT slide structure (13 slides)
- 10 toughest viva questions with answers

---

## Final Metrics (from `reports/metrics.json`)

Training dataset: 462 curated samples (Positive: 183, Negative: 184, Neutral: 95)

| Model | CV Accuracy | CV Macro-F1 | Held-out Accuracy | Held-out Macro-F1 |
|-------|:-----------:|:-----------:|:-----------------:|:-----------------:|
| VADER | — | — | **92.64%** | **91.71%** |
| Logistic Regression | 91.1 ± 1.7% | 90.3 ± 2.0% | 91.40% | 89.79% |
| Naive Bayes | 92.9 ± 2.4% | 93.1 ± 2.7% | 95.70% | 95.61% |
| RoBERTa | — | — | ~72% (TweetEval benchmark) | ~72% |

> **Regenerate:** `python scripts/evaluate.py`

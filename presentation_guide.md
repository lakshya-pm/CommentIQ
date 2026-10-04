# CommentIQ — Project Readiness & Presentation Guide
### With Exact File Locations & Code Line Numbers

---

## ✅ Project Readiness Assessment (vs Faculty Constraints)

| # | Constraint | Status | Notes |
|---|-----------|--------|-------|
| 1 | Complete pipeline | ✅ Ready | Scraping → Preprocessing → Training → Evaluation → Output |
| 2 | Web Scraping | ✅ Ready | YouTube Data API v3, paginated, up to 5000 comments |
| 3 | Preprocessing | ✅ Ready | 8-step NLTK pipeline (lowercase → lemmatize) |
| 4 | Model Training | ✅ Ready | VADER + LR + NB + RoBERTa, params in `sentiment.py` |
| 5 | Base Paper | ⚠️ Study needed | VADER (Hutto & Gilbert 2014) + TweetEval (Barbieri 2020) |
| 6 | Literature Survey | ⚠️ Study needed | See Literature Survey section below |
| 7 | Best Model | ✅ Ready | Logistic Regression (90%+ acc) — explain why |
| 8 | Results | ✅ Ready | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 9 | PPT | ⚠️ Create needed | Use outline at end of this guide |
| 10 | HuggingFace Models | ✅ Added | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| 11 | Everyone knows PPT | ⚠️ Team prep needed | Study this guide |
| 12 | Own Model | ✅ Ready | Each member assigned a model below |
| 13 | LoRA / LLM fine-tuning | ⚠️ Theory needed | See LoRA section below |
| 14 | .pkl model file | ✅ Added | Auto-saved to `downloads/models/` after first analysis |

---

## 🗂️ Project File Structure

```
youtube-comment-sentiment-analyzer/
├── app.py                          ← Flask app, route orchestrator
├── config.py                       ← API key loading from .env
├── modules/
│   ├── youtube.py                  ← Stage 1: Data collection
│   ├── preprocessing.py            ← Stage 2: NLP preprocessing
│   ├── training_data.py            ← Stage 3: Training dataset (497 samples)
│   ├── sentiment.py                ← Stage 3-5: Models + evaluation
│   ├── transformer.py              ← Stage 4: HuggingFace RoBERTa
│   └── visualization.py           ← Stage 6: Plotly charts
├── templates/
│   ├── index.html                  ← Home page (URL input form)
│   └── dashboard.html             ← Results dashboard
├── static/css/style.css           ← UI styling
├── downloads/
│   ├── models/                    ← .pkl saved models
│   └── result.csv                 ← Exported results
└── requirements.txt
```

---

## 🔁 Complete Pipeline with Exact Code Locations

```
YouTube URL
    ↓
[Stage 1] WEB SCRAPING
    ↓
[Stage 2] PREPROCESSING
    ↓
[Stage 3] FEATURE EXTRACTION + TRAINING
    ↓
[Stage 4] SENTIMENT MODELS (4 models)
    ↓
[Stage 5] EVALUATION
    ↓
[Stage 6] VISUALIZATION & OUTPUT
```

---

## 📌 Stage-by-Stage Answers — With Exact File/Line Numbers

---

### Stage 1 — Web Scraping

**File:** [`modules/youtube.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/youtube.py)

| What | Code Location |
|------|--------------|
| YouTube API client setup | [`youtube.py:41–59`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/youtube.py#L41-L59) — `get_youtube_client()` |
| Extract video ID from URL | [`youtube.py:66–202`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/youtube.py#L66-L202) — `extract_video_id()` |
| Fetch video metadata | [`youtube.py:250–383`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/youtube.py#L250-L383) — `fetch_video_details()` |
| Fetch comments (paginated) | [`youtube.py:390–485`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/youtube.py#L390-L485) — `fetch_comments(video_id, limit=500)` |
| Main entry point | [`youtube.py:492–562`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/youtube.py#L492-L562) — `fetch_video_from_url(url, limit)` |
| Called from Flask | [`app.py:92`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/app.py#L92) — `data = fetch_video_from_url(url, limit=comment_limit)` |

**Q: What data is collected?**
- Comment text (top-level comments only, up to 5,000)
- Video metadata: title, channel, views, likes, comment count, thumbnail, published date
- YouTube API returns 100 comments per page — paginated with `nextPageToken`

**Q: API quota cost?**
- Each `commentThreads.list` call = 1 unit
- Daily free limit = 10,000 units
- 500 comments = 5 units (negligible)

---

### Stage 2 — Preprocessing

**File:** [`modules/preprocessing.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py)

| Step | Function | Line |
|------|---------|------|
| Lowercase | `text.lower()` | ~[`L215`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L215) |
| Remove URLs | `remove_urls(text)` | ~[`L218`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L218) |
| Remove @mentions | `remove_mentions(text)` | ~[`L221`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L221) |
| Remove #hashtags | `remove_hashtags(text)` | ~[`L224`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L224) |
| Remove punctuation | `remove_punctuation(text)` | ~[`L227`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L227) |
| Tokenization | `tokenize(text)` — `word_tokenize` | ~[`L233`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L233) |
| Stopword removal | `remove_stopwords(tokens)` | ~[`L236`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L236) |
| Lemmatization | `lemmatize(tokens)` — `WordNetLemmatizer` | ~[`L239`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/preprocessing.py#L239) |
| Full pipeline entry | `preprocess_text(text)` | ~`L210` |
| Pipeline demo (for UI) | `get_preprocessing_steps(text)` | ~`L253` |
| Called from sentiment | [`sentiment.py:459`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L459) — `preprocess_text(t)` for each training sample |

**Q: What does TF-IDF take as input?**
> The preprocessed text (output of `preprocess_text()`) — not the original raw text.

**Q: Does RoBERTa use preprocessed text?**
> No — `transformer.py` receives `original_text` (raw), because transformers have their own internal tokenizer.
> See [`sentiment.py:L503`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L503) — `original_texts = df["original_text"].tolist()`

---

### Stage 3 — Training Dataset & Feature Extraction

**File:** [`modules/training_data.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/training_data.py)

| What | Location |
|------|----------|
| `TRAINING_DATA` list | [`training_data.py:L21`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/training_data.py#L21) |
| Positive samples (200) | [`training_data.py:L27–L221`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/training_data.py#L27-L221) |
| Negative samples (200) | [`training_data.py:L225–L419`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/training_data.py#L225-L419) |
| Neutral samples (97) | [`training_data.py:L423–L500`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/training_data.py#L423-L500) |
| NLTK corpus accessor | [`training_data.py:get_nltk_corpus()`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/training_data.py) — reference only, not used for training |

**Dataset stats:**
- Total: **497 samples** (200 Pos / 200 Neg / 97 Neutral)
- Train/Test split: 80/20 stratified
- Train: ~397 samples | Test: ~100 samples

**File:** [`modules/sentiment.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py)

| What | Location |
|------|----------|
| `FeatureExtractor` class (TF-IDF) | [`sentiment.py:L148–L190`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L148-L190) |
| TF-IDF parameters | [`sentiment.py:L158–L165`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L158-L165) |

```python
# sentiment.py:158-165
TfidfVectorizer(
    max_features = 8000,     # vocabulary size cap
    ngram_range  = (1, 2),   # unigrams + bigrams
    min_df       = 1,        # include even rare terms
    max_df       = 0.95,     # ignore words in >95% of docs
    sublinear_tf = True      # use log(1 + tf) scaling
)
```

---

### Stage 4 — Models (4 Models)

**File:** [`modules/sentiment.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py)

#### Model 1: VADER

| What | Location |
|------|----------|
| Import + init | [`sentiment.py:L38, L67`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L38) |
| VADER analysis per comment | [`sentiment.py:L90–L135`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L90-L135) — `analyze_comment()` |
| Threshold logic | compound ≥ 0.05 → Positive, ≤ -0.05 → Negative, else Neutral |

#### Model 2: Logistic Regression ⭐ (Best Model)

| What | Location |
|------|----------|
| Class definition | [`sentiment.py:L200–L215`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L200-L215) — `SentimentModels.__init__()` |
| LR parameters | [`sentiment.py:L205–L211`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L205-L211) |
| Training | [`sentiment.py:L252–L262`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L252-L262) |
| Save to .pkl | [`sentiment.py:L255–L261`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L255-L261) — `joblib.dump()` |

```python
# sentiment.py:205-211
LogisticRegression(
    C        = 5.0,    # inverse regularization strength
    max_iter = 2000,   # convergence iterations
    solver   = 'lbfgs' # Limited-memory BFGS optimizer
)
```

**Results:** Accuracy **90%** | Precision **90.08%** | Recall **90%** | F1 **89.99%**
**Saved as:** `downloads/models/logistic_regression.pkl`

#### Model 3: Multinomial Naive Bayes

| What | Location |
|------|----------|
| NB parameters | [`sentiment.py:L213–L215`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L213-L215) |
| Training | [`sentiment.py:L264–L274`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L264-L274) |
| Save to .pkl | `joblib.dump()` to `downloads/models/naive_bayes.pkl` |

```python
MultinomialNB(alpha=0.1)  # Laplace smoothing = 0.1
```

**Results:** Accuracy **89%** | F1 **88.96%**
**Saved as:** `downloads/models/naive_bayes.pkl`

#### Model 4: HuggingFace RoBERTa 🤗

**File:** [`modules/transformer.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/transformer.py)

| What | Location |
|------|----------|
| Model ID | [`transformer.py:L33`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/transformer.py#L33) — `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Pipeline init (lazy) | [`transformer.py:L54–L81`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/transformer.py#L54-L81) — `get_transformer_pipeline()` |
| Batch inference | [`transformer.py:L84–L126`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/transformer.py#L84-L126) — `transformer_predict(texts, batch_size=32)` |
| Called from sentiment | [`sentiment.py:L499–L515`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L499-L515) |
| Cache location | `downloads/hf_cache/cardiffnlp/` |

**Q: Why this model and NOT HingRoBERT?**
> `l3cube-pune/hing-roberta` is a **base pre-trained model** (MLM only) — it has **no classification head** and cannot output sentiment labels without full fine-tuning on a labeled dataset.
>
> `cardiffnlp/twitter-roberta-base-sentiment-latest` is **already fine-tuned** for 3-class sentiment (Neg/Neu/Pos) on the TweetEval benchmark (same social media domain as YouTube comments).

**Architecture:**
- RoBERTa-base: 12 transformer layers, 768 hidden dim, 12 attention heads
- 125M total parameters
- Pre-trained: 58M tweets (masked language modeling)
- Fine-tuned: TweetEval sentiment benchmark

---

### Stage 5 — Evaluation

**File:** [`modules/sentiment.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py)

| What | Location |
|------|----------|
| Evaluation method | [`sentiment.py:L290–L370`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L290-L370) — `SentimentModels._evaluate()` |
| Train/test split (80/20) | [`sentiment.py:L248–L252`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/sentiment.py#L248-L252) — `train_test_split(X, y, test_size=0.2, stratify=y)` |
| Metrics computed | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix` |
| Results stored | `models.evaluation_results` dict |

**Q: Explain each metric**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP+TN) / Total | % of all predictions correct |
| **Precision** | TP / (TP+FP) | Of predicted Positives, % that are correct |
| **Recall** | TP / (TP+FN) | Of actual Positives, % that were found |
| **F1-Score** | 2×P×R / (P+R) | Harmonic mean — balances precision & recall |
| **Confusion Matrix** | Grid of TP/FP/TN/FN | Shows which classes get confused |

**Q: Why weighted average?**
> Class imbalance: YouTube comments skew positive. Weighted avg accounts for class size.

---

### Stage 6 — Visualization

**File:** [`modules/visualization.py`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/modules/visualization.py)
**Template:** [`templates/dashboard.html`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/templates/dashboard.html)

| Chart | Function |
|-------|---------|
| Sentiment Pie Chart | `create_pie_chart(summary)` |
| Sentiment Bar Chart | `create_sentiment_bar_chart(summary)` |
| Compound Score Distribution | `create_compound_distribution(df)` |
| Word Cloud | `create_wordcloud(df)` → saved to `static/generated/wordcloud.png` |
| Top Positive Words | `create_positive_chart(df)` |
| Top Negative Words | `create_negative_chart(df)` |
| Confusion Matrix | `create_confusion_matrix_chart(ml_metrics, model)` |
| Model Comparison | `create_model_comparison_chart(ml_metrics)` |

All charts use **Plotly** (interactive, JSON-serialized, rendered in browser via `Plotly.newPlot()`).
Called from [`app.py:L143–L153`](file:///c:/Users/begre/OneDrive/Documents/nlp%20project/youtube-comment-sentiment-analyzer/app.py#L143-L153).

---

## 💾 .pkl Model Files

After running one analysis, these files are created automatically:

```
downloads/
└── models/
    ├── logistic_regression.pkl   ← Best model ⭐
    ├── naive_bayes.pkl
    └── tfidf_vectorizer.pkl      ← Required to use LR/NB
```

**To load and use the saved model (for demo):**
```python
import joblib

vectorizer = joblib.load("downloads/models/tfidf_vectorizer.pkl")
model      = joblib.load("downloads/models/logistic_regression.pkl")

X = vectorizer.transform(["This video is absolutely amazing"])
pred = model.predict(X)
print(pred)   # → ['Positive']
```

> [!IMPORTANT]
> Run at least one video analysis in the app BEFORE the presentation to ensure .pkl files exist.

---

## 📄 Base Papers to Study

### Paper 1 (for VADER model):
**"VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text"**
- Authors: C.J. Hutto, Eric Gilbert
- Venue: ICWSM 2014
- Dataset: Twitter, Amazon, Movie Reviews
- Key result: Outperforms LIWC, ANEW, SentiWordNet on social media
- **Our difference:** We add TF-IDF + ML + Transformer on top of VADER for better 3-class accuracy

### Paper 2 (for RoBERTa model):
**"TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification"**
- Authors: Barbieri et al.
- Venue: EMNLP Findings 2020
- Model used: `cardiffnlp/twitter-roberta-base-sentiment-latest` (this is the paper's model)
- F1 on sentiment: 72.0

---

## 📚 Literature Survey Table

| Paper | Model | Dataset | Result | Limitation |
|-------|-------|---------|--------|------------|
| Hutto & Gilbert (2014) | VADER | Twitter/Amazon | Best on social media | No learning |
| Kim (2014) | TextCNN | SST-2 | 88.1% acc | Binary only |
| Devlin et al. (2018) | BERT | 11 benchmarks | SOTA across NLP | Heavy compute |
| Liu et al. (2019) | RoBERTa | Multiple | Beats BERT | Still heavy |
| Barbieri et al. (2020) | Twitter-RoBERTa | TweetEval | 72.0 F1 | English only |
| Joshi et al. (2022) | HingBERT/HingRoBERTa | HingCorpus | SOTA Hinglish | No sentiment head |

---

## 🤖 LoRA & LLM Fine-Tuning (Constraint 13)

**Q: What is LoRA?**
> **Low-Rank Adaptation** — efficient fine-tuning for large language models.
> - Freeze original model weights (125M params)
> - Add small trainable matrices A (d×r) and B (r×k) to attention layers
> - Only train A and B: typically r=8 → ~0.5M trainable params (0.4% of total)
> - At inference: `W_adapted = W_original + B·A`

**Q: LoRA config for our use case:**
```python
from peft import get_peft_model, LoraConfig, TaskType

config = LoraConfig(
    task_type      = TaskType.SEQ_CLS,   # sequence classification
    r              = 8,                  # rank of adaptation matrices
    lora_alpha     = 32,                 # scaling factor (α/r = 4)
    lora_dropout   = 0.1,               # dropout on LoRA layers
    target_modules = ["query", "value"] # which attention layers to adapt
)
model = get_peft_model(hing_roberta_model, config)
# Trainable params: ~600K out of 125M (0.48%)
```

**Q: How is LoRA different from full fine-tuning?**

| | Full Fine-tuning | LoRA |
|--|--|--|
| Params updated | 125M (all) | ~600K (0.5%) |
| GPU RAM needed | ~4GB | ~500MB |
| Training time | Hours | Minutes |
| Risk | Catastrophic forgetting | Minimal |
| Performance | Best | Near-best |

---

## 👥 Team Member Assignments

| Member | Owns | Must Know |
|--------|------|-----------|
| Member 1 | VADER | Lexicon approach, compound score formula, thresholds |
| Member 2 | Logistic Regression | TF-IDF, LR math, C parameter, .pkl saving, `sentiment.py:L205–L261` |
| Member 3 | Naive Bayes | Bayes theorem, conditional probability, alpha smoothing, `sentiment.py:L213–L274` |
| Member 4 | HuggingFace RoBERTa | Transformer architecture, attention mechanism, why cardiffnlp > HingRoBERT, LoRA theory, `transformer.py` |

---

## 🖥️ PPT Slide Outline

1. **Title** — CommentIQ: YouTube Comment Sentiment Analysis
2. **Problem Statement** — Why analyze YouTube comments?
3. **System Architecture** — Pipeline diagram (6 stages)
4. **Stage 1: Web Scraping** — YouTube Data API v3, paginated 100/page, up to 5000
5. **Stage 2: Preprocessing** — 8-step NLTK pipeline table
6. **Stage 3: Feature Extraction** — TF-IDF explanation + parameters
7. **Model 1: VADER** — Rule-based, compound score, thresholds
8. **Model 2: Logistic Regression** — TF-IDF + supervised ML, 90% acc ⭐
9. **Model 3: Naive Bayes** — Probabilistic, Bayes theorem, 89% acc
10. **Model 4: HuggingFace RoBERTa** — Transformer, Twitter fine-tuned, why not HingRoBERT
11. **Training Dataset** — 497 curated samples, 80/20 split, distribution
12. **Results Table** — LR vs NB vs RoBERTa comparison
13. **Confusion Matrix** — show from dashboard
14. **Base Paper** — VADER 2014 + TweetEval 2020
15. **Literature Survey** — Table of 6 papers
16. **LoRA / LLM Fine-tuning** — Theory, config, vs full fine-tuning
17. **Live Demo** — Open CommentIQ on a YouTube URL
18. **Model .pkl Files** — Show `downloads/models/` folder
19. **Conclusion** — Contributions + future: LoRA fine-tuning HingRoBERT on our dataset

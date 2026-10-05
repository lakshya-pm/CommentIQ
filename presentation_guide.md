# CommentIQ — Complete Presentation & Viva Guide

> Source of truth for the live demo and viva.
> Every number here comes from `python scripts/evaluate.py` — never hardcoded.

---

## 0. Quick Reference — Numbers to Know Cold

| Model | CV Accuracy | CV Macro-F1 | Held-out Acc |
|-------|:-----------:|:-----------:|:------------:|
| VADER | N/A | N/A | **92.64%** |
| Logistic Regression | **91.1 ± 1.7%** | 90.3 ± 2.0% | 91.40% |
| Naive Bayes | **92.9 ± 2.4%** | 93.1 ± 2.7% | 95.70% |
| RoBERTa | — | — | ~72% (TweetEval) |

Training set: **462 curated samples** · Test: stratified 5-fold CV + 20% held-out

> Regenerate anytime: `python scripts/evaluate.py`

---

## 1. The Pipeline (Know This End-to-End)

```
URL Input
   ↓
[youtube.py / reddit.py]   ← API calls with limit parameter
   ↓
Raw comment list
   ↓
[preprocessing.py]         ← 8 steps, negations KEPT
   ↓
Cleaned text
   ├──→ [VADER]            ← Rule-based, compound score
   ├──→ [TF-IDF → LR/NB]  ← Trained on 462 curated samples
   └──→ [RoBERTa]          ← cardiffnlp/twitter-roberta (HuggingFace)
   ↓
[Ensemble Vote]            ← Majority; RoBERTa breaks ties
   ↓
Dashboard (Plotly + word cloud + CSV download)
```

**File map:**

| Stage | File | Key lines |
|-------|------|-----------|
| Web scraping | `modules/youtube.py` | `fetch_video_from_url()` |
| Web scraping | `modules/reddit.py` | `fetch_post_from_url()` |
| Preprocessing | `modules/preprocessing.py` | `preprocess_text()` L1-end |
| VADER | `modules/sentiment.py` | `classify_with_vader()` L135 |
| TF-IDF | `modules/sentiment.py` | `FeatureExtractor` class L174 |
| LR/NB training | `modules/sentiment.py` | `SentimentModels.train()` L236 |
| Model caching | `modules/sentiment.py` | `_ensure_models_loaded()` L449 |
| RoBERTa | `modules/transformer.py` | `transformer_predict()` |
| Ensemble | `modules/sentiment.py` | `_ensemble_vote()` L569 |
| Flask routes | `app.py` | `/analyze` route |
| Metrics script | `scripts/evaluate.py` | Source of truth for all numbers |

---

## 2. Web Scraping — What to Say

### Source
- **YouTube**: YouTube Data API v3 via `google-api-python-client`
- **Reddit**: Reddit PRAW API

### What we collect
- YouTube: comment text, video title, channel, views, likes, thumbnail, published date
- Reddit: post title, subreddit, comment text, author, score

### Key design decisions
- Default **500 comments** (configurable via `comment_limit` in the form)
- **Why 500?** API quota — YouTube Data API v3 gives 10,000 units/day; one comment page costs 1 unit, each video can have thousands of comments. 500 gives a representative sample without burning quota.
- Each request uses a **UUID-named CSV file** to avoid concurrent request races (e.g., `downloads/results/abc-123.csv`)

### Viva Q&A
**Q: Why not scrape all comments?**
> YouTube API quota is 10,000 units/day. Each 100-comment page costs ~1 unit. Fetching all comments on a popular video (100K+ comments) would exhaust the daily quota. 500 is a good tradeoff — statistically representative with <5% error margin for proportions.

**Q: Why use the official API and not scrape HTML?**
> The API is stable, rate-limited properly, and compliant with YouTube ToS. HTML scraping breaks every time YouTube changes its layout.

---

## 3. Preprocessing — Know Every Step

### The 8-step pipeline in `modules/preprocessing.py → preprocess_text()`

| Step | What it does | Why |
|------|-------------|-----|
| 1. Null check | Returns `""` for None/empty | Prevents downstream crashes |
| 2. Contraction expansion | `can't → cannot`, `won't → will not` | Must happen BEFORE punctuation removal |
| 3. Lowercase | `Good → good` | Normalise case |
| 4. URL removal | `http://...` stripped | URLs add no sentiment signal |
| 5. Mention removal | `@user` stripped | Irrelevant to content |
| 6. Hashtag → word | `#great → great` | Keep the word, remove the symbol |
| 7. Punctuation & numbers | Non-alpha stripped | Reduce noise |
| 8. Stopword removal (negations KEPT) | `the, is, a` removed but `not, no, never, nor` kept | Critical fix — old code deleted "not" making "not good" → "good" |
| 9. Lemmatization | `running → run` | Normalise verb forms |

### The negation bug (important for viva)
> Old NLTK stopword list includes "not", "no", "nor". So `"This is not good"` became `"good"` — positive sentiment. We fixed this by explicitly removing negation words from the stopword set AFTER contraction expansion (so `can't → cannot → not` is preserved).

### Viva Q&A
**Q: Why expand contractions before punctuation removal?**
> If we remove punctuation first, `can't` becomes `cant` (not a valid word). Expanding first gives `cannot`, which preserves the negation correctly.

**Q: What's lemmatization vs stemming?**
> Lemmatization uses vocabulary and grammar to return the base dictionary form (`better → good`, `running → run`). Stemming just chops suffixes (`running → runn`). Lemmatization is more accurate but slower.

---

## 4. Models — Know All Four

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Rule-based lexicon
- **Input**: Original text (not preprocessed — VADER has its own rules)
- **How**: Compound score from -1 to +1. ≥0.05 → Positive, ≤-0.05 → Negative
- **Strength**: Fast, handles emojis, slang, ALL CAPS, punctuation emphasis
- **Paper**: Hutto & Gilbert, ICWSM 2014

### Logistic Regression
- **Type**: Supervised ML, linear classifier
- **Input**: TF-IDF features (5000 max, unigrams + bigrams)
- **Training**: 462 curated samples, `C=5.0`, `lbfgs` solver
- **CV accuracy**: **91.1 ± 1.7%**
- **Why**: Interpretable, fast, good baseline for text classification

### Naive Bayes (Multinomial)
- **Type**: Probabilistic classifier, assumes feature independence
- **Input**: TF-IDF features
- **Training**: Same 462 samples, `alpha=0.1` (Laplace smoothing)
- **CV accuracy**: **92.9 ± 2.4%**
- **Why**: Works well with TF-IDF, handles sparse matrices naturally

### RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Type**: Transformer, 125M parameters
- **Input**: Raw original text (own BPE tokenizer, handles emojis)
- **Pre-training**: 124M tweets from TimeLMs project (Loureiro et al. 2022)
- **Fine-tuning**: Twitter Sentiment (TweetEval) benchmark
- **Why this model**: Pre-trained on social-media text including YouTube-like language; has a built-in sentiment classification head
- **Benchmark**: ~72% macro-F1 on TweetEval (3-class sentiment)

### Viva Q&A
**Q: Why use four models instead of just RoBERTa?**
> RoBERTa is slow (~2s per batch) and requires GPU for speed. LR/NB are instant (<10ms). VADER is completely offline. The ensemble combines speed with accuracy. Also, using multiple approaches is a richer academic demonstration.

**Q: Why does Naive Bayes sometimes outperform Logistic Regression on the test set?**
> NB with small, well-curated datasets often outperforms LR because the independence assumption is approximately satisfied in clean training data. NB is also more robust to the small sample sizes (462 examples).

**Q: What is the domain gap?**
> LR/NB are trained on 462 clean curated sentences. Real YouTube comments use slang, abbreviations, emoji-only comments, and non-English text that the TF-IDF vocabulary has never seen. About 30-40% of live comments get zero feature overlap — those are flagged "Low confidence" and deferred to RoBERTa.

---

## 5. Base Paper & Literature

### Base Paper
**VADER** — Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. ICWSM.

**Key points to know:**
- VADER was designed specifically for social media (Twitter, Facebook)
- Lexicon of 7500+ labeled features with valence scores
- 5 heuristics: punctuation, capitalization, degree modifiers, conjunctions, tri-gram negation
- 82.9% accuracy on Twitter data (our VADER scores 92.64% on the curated set)

### RoBERTa Source Paper
**TimeLMs** — Loureiro et al. (2022). *TimeLMs: Diachronic Language Models from Twitter*. ACL Findings.

**TweetEval** — Barbieri et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. EMNLP Findings.

### Literature Survey — Key Points

| Paper | Model | Dataset | Accuracy | Limitation |
|-------|-------|---------|----------|-----------|
| Hutto & Gilbert 2014 | VADER | Twitter | 82.9% | Rule-based, misses context |
| Barbieri et al. 2020 | RoBERTa | TweetEval | 72.6% | Social media domain only |
| Zhang et al. 2018 | BERT | SST-2 | 94.9% | Requires labelled in-domain data |
| This project | Ensemble | Curated 462 | 92.64% (VADER), 91.1% (LR), 92.9% (NB) | Domain gap on live data |

---

## 6. Evaluation — Know Every Metric

### Why 5-fold CV (not single split)?
> A single 80/20 split on 462 samples gives 93 test samples. One lucky split can show 94%+. 5-fold CV uses all data for testing across 5 runs — the mean ± standard deviation is a much more reliable estimate.

### Metrics used

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Accuracy | correct / total | Overall fraction correct |
| Precision | TP / (TP+FP) | Of predicted Positive, how many actually are |
| Recall | TP / (TP+FN) | Of actual Positive, how many we caught |
| Macro-F1 | Mean of per-class F1 | Balanced across all 3 classes |
| Weighted-F1 | Class-size-weighted F1 | Favours majority class |

**Which to report**: **Macro-F1 is most important** for imbalanced classes (Neutral is underrepresented at 95/462 ≈ 21%). Accuracy alone is misleading when class sizes differ.

### Our reported numbers (from `reports/metrics.json`)
- **VADER**: 92.64% accuracy · 91.71% macro-F1
- **LR**: 91.1 ± 1.7% CV accuracy · 90.3% macro-F1
- **NB**: 92.9 ± 2.4% CV accuracy · 93.1% macro-F1

---

## 7. Known Limitations — Be Proactive, Not Defensive

State these before the examiner asks:

1. **Domain gap**: LR/NB vocabulary trained on 462 curated sentences; ~30-40% of live YouTube comments have zero feature overlap. Mitigated by the "Low confidence" flag + RoBERTa fallback.

2. **Small training set**: 462 samples is small. 5-fold CV addresses the single-split inflation but doesn't eliminate overfitting risk.

3. **Evaluation mismatch**: Metrics are on the curated test set, not on live YouTube comments. Real-world accuracy is likely lower — we don't have ground truth for live data.

4. **Language**: English only. Non-English comments are processed but results are unreliable.

5. **Sarcasm**: All models struggle with sarcasm (e.g., "Oh great, another terrible video" scores Positive on surface words).

6. **RoBERTa speed**: ~2s per batch on CPU, making 500 comments take ~15-20 seconds. GPU would reduce this to <1s.

---

## 8. Live Demo Script

### Setup (before the demo)
```bash
# Make sure Flask is running
python app.py

# Have these URLs ready in notepad:
# YouTube (mixed sentiment): https://www.youtube.com/watch?v=dQw4w9WgXcQ
# Reddit (tech discussion):   https://www.reddit.com/r/MachineLearning/
```

### During the demo

1. **Open** `http://127.0.0.1:5000`
2. **Paste** the YouTube URL, set limit to 100
3. **Click Analyze** — while loading, explain the pipeline
4. **Show the dashboard**:
   - Point to the pie chart: "This is the ensemble label — majority vote of 4 models"
   - Point to the model agreement table: "VADER and RoBERTa agree most often; LR/NB show 'Low confidence' for comments outside their vocabulary"
   - Point to the word cloud: "Sized by frequency in the comment section"
5. **Click Download CSV** — show the per-comment breakdown with all 4 model labels

### If the API quota is exceeded
```bash
# In .env:
DEMO_MODE=true
# Restart Flask — shows bundled sample result without making API calls
```

---

## 9. PPT Slide Structure

| Slide | Content |
|-------|---------|
| 1 | Title — CommentIQ, Author, Course |
| 2 | Problem Statement + motivation (YouTube has 500M+ comments/day) |
| 3 | Pipeline diagram (from preprocessing.py → dashboard) |
| 4 | Web Scraping — sources, API, data collected |
| 5 | Preprocessing — 8 steps, negation fix (show before/after) |
| 6 | Models Overview — VADER, LR, NB, RoBERTa in one table |
| 7 | Training — 462 curated samples, 5-fold CV, TF-IDF config |
| 8 | Results table — all 4 models, accuracy + macro-F1 |
| 9 | Confusion matrices (from `reports/`) |
| 10 | Ensemble voting — diagram of how majority vote works |
| 11 | Limitations — domain gap, small training set, sarcasm |
| 12 | Live Demo → switch to browser |
| 13 | Conclusion + References (VADER paper, TimeLMs, TweetEval) |

**PPT rules from faculty:**
- Content concise, not overcrowded
- Simple language, clear graphs
- Don't read from slides
- Every metric slide → come from `reports/metrics.json`

---

## 10. Viva Cheat Sheet — Toughest Questions

**Q: Why 462 samples specifically?**
> It's the complete curated training set from `modules/training_data.py`. Adding more samples requires careful curation — random internet data (like the Kaggle YouTube dataset we evaluated) has ~20% label noise that drops accuracy to 60%. Quality > quantity for supervised learning on small datasets.

**Q: What is TF-IDF?**
> Term Frequency-Inverse Document Frequency. TF = how often a word appears in one comment. IDF = log(total comments / comments containing the word). Words that appear in every comment (like "the") get low IDF; rare discriminative words get high IDF. We use unigrams + bigrams (up to 5000 features).

**Q: Why use an ensemble?**
> Each model has complementary strengths. VADER handles slang and emojis. LR/NB are fast and trained on curated examples. RoBERTa understands context and word order. Majority voting reduces individual model errors — empirically, ensemble accuracy is higher than any single model.

**Q: What is RoBERTa?**
> Robustly Optimized BERT Approach. A transformer architecture pre-trained on 160GB of text via masked language modeling. We use the `cardiffnlp/twitter-roberta-base-sentiment-latest` variant fine-tuned on 124M tweets — directly relevant to our social-media use case.

**Q: What would you improve with more time?**
> 1. In-domain training data — manually label 2000+ YouTube comments for LR/NB training. 2. LoRA fine-tune RoBERTa on those labels (script already written: `scripts/finetune_lora.py`). 3. Multi-language support via `multilingual-sentiment-analysis` models.

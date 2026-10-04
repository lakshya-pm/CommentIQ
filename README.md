# 🚀 CommentIQ — NLP Sentiment Analysis Application

CommentIQ is a Flask-based web application that performs **sentiment analysis** on comments from **YouTube videos** and **Reddit posts** using a complete **Natural Language Processing (NLP)** pipeline. It applies **Machine Learning** classification and presents results through **interactive visualizations**.

---

## 📌 Project Overview

This project demonstrates a complete **Sentiment Analysis Workflow** for social media text data:

### Workflow of Sentiment Analysis Application

1. **Data Collection**
   - Load comments from YouTube (Data API v3) or Reddit (PRAW API)

2. **Text Preprocessing**
   - Tokenization
   - Lowercasing
   - Removing stopwords, URLs, mentions (@user), hashtags (#tag), punctuation
   - Lemmatization

3. **Feature Extraction**
   - Convert text into numerical format using:
     - TF-IDF (Term Frequency–Inverse Document Frequency)
     - N-grams (unigrams + bigrams)

4. **Model Building**
   - Use Machine Learning algorithms:
     - **VADER** (primary sentiment classifier)
     - **Logistic Regression**
     - **Naive Bayes**

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score, Confusion Matrix

6. **Prediction & Visualization**
   - Predict sentiment for comments
   - Display results using pie chart, word cloud, bar graphs

---

## 📊 Features

- 🔍 Analyze YouTube video comments (including Shorts)
- 💬 Analyze Reddit post comments
- 🧹 Full NLP text preprocessing pipeline
- 📐 TF-IDF feature extraction with N-grams
- 🤖 VADER + Logistic Regression + Naive Bayes classification
- 📈 Model evaluation metrics (Accuracy, Precision, Recall, F1)
- 📊 Interactive Plotly charts (Pie, Bar, Distribution, Confusion Matrix)
- ☁️ Word Cloud generation
- 🔬 Preprocessing pipeline visualization
- 📥 Download analysis results as CSV
- 🌐 Modern, responsive web interface

---

## 🛠️ Tech Stack

### Frontend
- HTML5, CSS3 (custom design system)
- JavaScript (vanilla)
- Plotly.js (interactive charts)

### Backend
- Python, Flask

### NLP & Data Processing
- **NLTK** — Tokenization, stopwords, lemmatization, VADER
- **Scikit-learn** — TF-IDF, Logistic Regression, Naive Bayes, metrics
- **Pandas, NumPy** — Data handling

### Visualization
- **Plotly** — Interactive charts
- **Matplotlib** — Word cloud generation
- **WordCloud** — Word cloud library

### APIs
- YouTube Data API v3
- Reddit API (PRAW)

---

## 📂 Project Structure

```
CommentIQ/
│
├── app.py                    # Flask application (main entry)
├── config.py                 # API keys configuration
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
│
├── modules/
│   ├── __init__.py
│   ├── youtube.py            # YouTube API integration
│   ├── reddit.py             # Reddit API integration
│   ├── preprocessing.py      # NLP text preprocessing pipeline
│   ├── sentiment.py          # Sentiment analysis + ML models
│   └── visualization.py      # Plotly & matplotlib visualizations
│
├── templates/
│   ├── index.html            # Home page
│   └── dashboard.html        # Analysis dashboard
│
├── static/
│   ├── css/style.css         # Custom stylesheet
│   ├── js/script.js          # Client-side JavaScript
│   └── generated/            # Generated charts (word cloud)
│
├── downloads/                # CSV export directory
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/youtube-comment-sentiment-analyzer.git
cd youtube-comment-sentiment-analyzer
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Data (automatic on first run)

The application automatically downloads required NLTK resources:
- `punkt`, `stopwords`, `wordnet`, `vader_lexicon`

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY

REDDIT_CLIENT_ID=YOUR_CLIENT_ID
REDDIT_CLIENT_SECRET=YOUR_CLIENT_SECRET
REDDIT_USER_AGENT=YOUR_USER_AGENT
```

---

## ▶️ Run the Project

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data handling |
| nltk, re | Text preprocessing |
| scikit-learn | ML model training, TF-IDF, metrics |
| matplotlib, wordcloud | Word cloud visualization |
| plotly | Interactive charts |
| Flask | Web framework |
| google-api-python-client | YouTube API |
| praw | Reddit API |

---

## 🌍 Applications of Sentiment Analysis

- Customer Feedback Monitoring
- Social Media Analytics
- Brand Management
- Political Sentiment Tracking
- Financial Market Prediction
- News Sentiment Classification

---

## 📄 References

1. **Kaggle Dataset**: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
2. **NLTK VADER**: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis
3. **Scikit-learn Documentation**: https://scikit-learn.org/

---

## 👨‍💻 Author

**Lakshya Marwaha**



---

## 📄 License

This project is developed for educational purposes as an **NLP Course Project**.

"""
===========================================================
CommentIQ - NLP Sentiment Analysis Application
===========================================================

Flask web application that performs sentiment analysis on
YouTube video comments and Reddit post comments using a
complete NLP pipeline.

NLP Pipeline Workflow:
1. Data Collection (YouTube API / Reddit API)
2. Text Preprocessing (Tokenization, Lowercasing,
   Stopword Removal, Lemmatization)
3. Feature Extraction (TF-IDF Vectorization)
4. Model Building (Logistic Regression + Naive Bayes)
5. Model Evaluation (Accuracy, Precision, Recall, F1)
6. Prediction & Visualization (Plotly Charts)

Author: Lakshya Marwaha
"""

from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os
import json

from modules.youtube import fetch_video_from_url
from modules.reddit import fetch_post_from_url

from modules.sentiment import (
    analyze_comments,
    sentiment_summary
)

from modules.visualization import (
    create_pie_chart,
    create_sentiment_bar_chart,
    create_compound_distribution,
    create_wordcloud,
    create_positive_chart,
    create_negative_chart,
    create_confusion_matrix_chart,
    create_model_comparison_chart
)

from modules.preprocessing import get_preprocessing_steps


app = Flask(__name__)

DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------
# Home Page
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------
# Analyze URL
# ---------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    url = request.form.get("url", "").strip()

    if not url:
        return render_template("index.html", error="Please enter a valid URL.")

    # Parse comment limit (user-selected)
    try:
        comment_limit = int(request.form.get("comment_limit", 500))
        comment_limit = max(100, min(comment_limit, 5000))
    except (ValueError, TypeError):
        comment_limit = 500

    try:

        # -----------------------------------------
        # Step 1: Data Collection
        # -----------------------------------------

        if "youtube.com" in url or "youtu.be" in url:

            platform = "YouTube"
            data = fetch_video_from_url(url, limit=comment_limit)

            title = data["title"]

            details = {
                "Channel": data["channel"],
                "Views": f"{data['views']:,}",
                "Likes": f"{data['likes']:,}",
                "Comments Fetched": f"{len(data['comments_list']):,}",
                "Published": data["published"]
            }

            thumbnail = data["thumbnail"]
            comments = data["comments_list"]

        elif "reddit.com" in url:

            platform = "Reddit"
            data = fetch_post_from_url(url)

            title = data["title"]

            details = {
                "Subreddit": f"r/{data['subreddit']}",
                "Author": f"u/{data['author']}",
                "Upvotes": f"{data['score']:,}",
                "Comments Fetched": f"{len(data['comments_list']):,}"
            }

            thumbnail = None
            comments = data["comments_list"]

        else:

            return render_template(
                "index.html",
                error="Unsupported URL. Please enter a YouTube or Reddit URL."
            )

        # -----------------------------------------
        # Steps 2-5: NLP Pipeline
        # -----------------------------------------

        df, ml_metrics = analyze_comments(comments)

        summary = sentiment_summary(df)

        # -----------------------------------------
        # Step 6: Visualization
        # -----------------------------------------

        pie_chart_json          = create_pie_chart(summary)
        bar_chart_json          = create_sentiment_bar_chart(summary)
        compound_chart_json     = create_compound_distribution(df)
        positive_words_json     = create_positive_chart(df)
        negative_words_json     = create_negative_chart(df)

        lr_confusion_json       = create_confusion_matrix_chart(ml_metrics, "logistic_regression")
        nb_confusion_json       = create_confusion_matrix_chart(ml_metrics, "naive_bayes")
        model_comparison_json   = create_model_comparison_chart(ml_metrics)

        create_wordcloud(df)

        # -----------------------------------------
        # Preprocessing Pipeline Demo
        # -----------------------------------------

        sample_comment    = comments[0] if comments else ""
        preprocessing_demo = get_preprocessing_steps(sample_comment)

        # -----------------------------------------
        # CSV Export
        # -----------------------------------------

        csv_path = os.path.join(DOWNLOAD_FOLDER, "result.csv")
        df.to_csv(csv_path, index=False)

        # -----------------------------------------
        # Top Comments by Category
        # -----------------------------------------

        text_col = "original_text" if "original_text" in df.columns else "text"

        positive_comments = df[df["sentiment"] == "Positive"].head(5).to_dict("records")
        negative_comments = df[df["sentiment"] == "Negative"].head(5).to_dict("records")
        neutral_comments  = df[df["sentiment"] == "Neutral"].head(5).to_dict("records")

        # -----------------------------------------
        # Render Dashboard
        # -----------------------------------------

        return render_template(
            "dashboard.html",
            platform=platform,
            title=title,
            thumbnail=thumbnail,
            details=details,
            summary=summary,
            pie_chart_json=pie_chart_json,
            bar_chart_json=bar_chart_json,
            compound_chart_json=compound_chart_json,
            positive_words_json=positive_words_json,
            negative_words_json=negative_words_json,
            ml_metrics=ml_metrics,
            lr_confusion_json=lr_confusion_json,
            nb_confusion_json=nb_confusion_json,
            model_comparison_json=model_comparison_json,
            positive_comments=positive_comments,
            negative_comments=negative_comments,
            neutral_comments=neutral_comments,
            preprocessing_demo=preprocessing_demo,
            text_col=text_col,
            total_analyzed=len(df)
        )

    except Exception as e:

        return render_template("index.html", error=str(e))


# ---------------------------------------------------
# Download CSV
# ---------------------------------------------------

@app.route("/download")
def download():

    csv_path = os.path.join(DOWNLOAD_FOLDER, "result.csv")

    if os.path.exists(csv_path):
        return send_file(
            csv_path,
            as_attachment=True,
            download_name="commentiq_sentiment_analysis.csv"
        )

    return "No results available", 404


# ---------------------------------------------------
# Run Application
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

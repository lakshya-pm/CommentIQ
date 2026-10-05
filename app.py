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
   Stopword Removal — negations kept, Lemmatization)
3. Feature Extraction (TF-IDF Vectorization)
4. Model Building (Logistic Regression + Naive Bayes)
5. Model Evaluation (Accuracy, Precision, Recall, F1)
6. Prediction & Visualization (Plotly Charts)

Author: Lakshya Marwaha
"""

import logging
import uuid
import os
import json

from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd

from modules.youtube import fetch_video_from_url
from modules.reddit import fetch_post_from_url

from modules.sentiment import (
    analyze_comments,
    sentiment_summary,
)

from modules.visualization import (
    create_pie_chart,
    create_sentiment_bar_chart,
    create_compound_distribution,
    create_wordcloud,
    create_positive_chart,
    create_negative_chart,
    create_confusion_matrix_chart,
    create_model_comparison_chart,
)

from modules.preprocessing import get_preprocessing_steps

# ---------------------------------------------------
# Logging — replaces bare print() throughout the app
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


app = Flask(__name__)

DOWNLOAD_FOLDER = "downloads"
RESULTS_FOLDER  = os.path.join(DOWNLOAD_FOLDER, "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)


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

    # Parse comment limit (user-selected dropdown)
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

            try:
                data = fetch_video_from_url(url, limit=comment_limit)
            except Exception as exc:
                err = str(exc)
                # Provide specific user-facing error messages
                if "quotaExceeded" in err or "quota" in err.lower():
                    return render_template(
                        "index.html",
                        error=(
                            "YouTube API daily quota exceeded. "
                            "Please try again tomorrow or use a different API key."
                        ),
                    )
                if "commentsDisabled" in err or "disabled" in err.lower():
                    return render_template(
                        "index.html",
                        error="Comments are disabled for this video.",
                    )
                if "forbidden" in err.lower() or "403" in err:
                    return render_template(
                        "index.html",
                        error="This video is private or age-restricted.",
                    )
                raise  # re-raise unexpected errors

            title     = data["title"]
            details   = {
                "Channel":           data["channel"],
                "Views":             f"{data['views']:,}",
                "Likes":             f"{data['likes']:,}",
                "Comments Fetched":  f"{len(data['comments_list']):,}",
                "Published":         data["published"],
            }
            thumbnail = data["thumbnail"]
            comments  = data["comments_list"]

        elif "reddit.com" in url:

            platform  = "Reddit"
            # Pass the user's comment limit to Reddit too (was hardcoded 100)
            data      = fetch_post_from_url(url, limit=comment_limit)
            title     = data["title"]
            details   = {
                "Subreddit":        f"r/{data['subreddit']}",
                "Author":           f"u/{data['author']}",
                "Upvotes":          f"{data['score']:,}",
                "Comments Fetched": f"{len(data['comments_list']):,}",
            }
            thumbnail = None
            comments  = data["comments_list"]

        else:
            return render_template(
                "index.html",
                error="Unsupported URL. Please enter a YouTube or Reddit URL.",
            )

        if not comments:
            return render_template(
                "index.html",
                error="No comments found for this URL.",
            )

        logger.info(
            "Analyzing %d comments for: %s", len(comments), title[:60]
        )

        # -----------------------------------------
        # Steps 2–5: NLP Pipeline
        # -----------------------------------------

        df, ml_metrics = analyze_comments(comments)

        summary = sentiment_summary(df)

        # -----------------------------------------
        # Step 6: Visualization
        # -----------------------------------------

        pie_chart_json        = create_pie_chart(summary)
        bar_chart_json        = create_sentiment_bar_chart(summary)
        compound_chart_json   = create_compound_distribution(df)
        positive_words_json   = create_positive_chart(df)
        negative_words_json   = create_negative_chart(df)
        lr_confusion_json     = create_confusion_matrix_chart(ml_metrics, "logistic_regression")
        nb_confusion_json     = create_confusion_matrix_chart(ml_metrics, "naive_bayes")
        model_comparison_json = create_model_comparison_chart(ml_metrics)

        create_wordcloud(df)

        # -----------------------------------------
        # Preprocessing Pipeline Demo (first comment)
        # -----------------------------------------

        sample_comment     = comments[0] if comments else ""
        preprocessing_demo = get_preprocessing_steps(sample_comment)

        # -----------------------------------------
        # CSV Export — per-request UUID (safe for concurrent users)
        # Shared downloads/result.csv caused race conditions.
        # -----------------------------------------

        result_id = str(uuid.uuid4())
        csv_path  = os.path.join(RESULTS_FOLDER, f"{result_id}.csv")
        df.to_csv(csv_path, index=False)

        # -----------------------------------------
        # Top Comments by Category
        # -----------------------------------------

        text_col = "original_text" if "original_text" in df.columns else "text"

        positive_comments = df[df["sentiment"] == "Positive"].head(5).to_dict("records")
        negative_comments = df[df["sentiment"] == "Negative"].head(5).to_dict("records")
        neutral_comments  = df[df["sentiment"] == "Neutral"].head(5).to_dict("records")

        # How many comments had zero TF-IDF features (domain gap metric)
        low_confidence_count = int(
            (df.get("lr_prediction", pd.Series(dtype=str)) == "Low confidence").sum()
        ) if "lr_prediction" in df.columns else 0

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
            total_analyzed=len(df),
            result_id=result_id,
            low_confidence_count=low_confidence_count,
        )

    except Exception as e:
        logger.exception("Error during analysis of URL: %s", url)
        return render_template("index.html", error=str(e))


# ---------------------------------------------------
# Download CSV — takes ?id= query param (UUID)
# ---------------------------------------------------

@app.route("/download")
def download():

    result_id = request.args.get("id", "").strip()

    # Security: make sure no path traversal
    if not result_id or "/" in result_id or "\\" in result_id or ".." in result_id:
        return "Invalid download ID.", 400

    csv_path = os.path.join(RESULTS_FOLDER, f"{result_id}.csv")

    if os.path.exists(csv_path):
        return send_file(
            csv_path,
            as_attachment=True,
            download_name="commentiq_sentiment_analysis.csv",
        )

    return "Result not found. Please run a new analysis.", 404


# ---------------------------------------------------
# Run Application
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

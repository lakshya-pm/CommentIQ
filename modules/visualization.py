"""
===========================================================
Visualization Module
CommentIQ - NLP Sentiment Analysis Project
===========================================================

Creates interactive Plotly visualizations and static charts:

✔ Interactive Pie Chart (Plotly)
✔ Interactive Bar Chart (Plotly)
✔ Word Cloud (matplotlib)
✔ Top Positive Words (Plotly)
✔ Top Negative Words (Plotly)
✔ Sentiment Distribution Bar Chart (Plotly)
✔ Compound Score Distribution (Plotly)
✔ Confusion Matrix Heatmap (Plotly)

All interactive charts return Plotly JSON for frontend rendering.
Static images are saved for fallback display.

Author: Lakshya Marwaha
"""

import os
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


# ---------------------------------------------------
# Output Directory
# ---------------------------------------------------

OUTPUT_DIR = os.path.join("static", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------
# Color Palette
# ---------------------------------------------------

COLORS = {
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral": "#f59e0b",
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "bg": "#ffffff",
    "text": "#1e293b",
    "gradient_1": "#6366f1",
    "gradient_2": "#8b5cf6",
    "gradient_3": "#a78bfa"
}


# ---------------------------------------------------
# Plotly Layout Template
# ---------------------------------------------------

def get_plotly_layout(title="", height=400):
    """Standard Plotly layout with consistent styling."""

    return go.Layout(
        title=dict(
            text=title,
            font=dict(
                size=18,
                color=COLORS["text"],
                family="Inter, sans-serif"
            ),
            x=0.5,
            xanchor="center"
        ),
        font=dict(
            family="Inter, sans-serif",
            color=COLORS["text"]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, sans-serif"
        )
    )


# ---------------------------------------------------
# Interactive Pie Chart
# ---------------------------------------------------

def create_pie_chart(summary):
    """
    Creates interactive Plotly pie chart for sentiment
    distribution.

    Returns Plotly JSON string.
    """

    labels = []
    values = []
    colors = []

    if summary.get("positive", 0) > 0:
        labels.append("Positive")
        values.append(summary["positive"])
        colors.append(COLORS["positive"])

    if summary.get("negative", 0) > 0:
        labels.append("Negative")
        values.append(summary["negative"])
        colors.append(COLORS["negative"])

    if summary.get("neutral", 0) > 0:
        labels.append("Neutral")
        values.append(summary["neutral"])
        colors.append(COLORS["neutral"])

    if not values:
        return json.dumps({})

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(
                    colors=colors,
                    line=dict(color="white", width=3)
                ),
                textinfo="label+percent",
                textfont=dict(size=14),
                hoverinfo="label+value+percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value}<br>"
                    "Percentage: %{percent}<extra></extra>"
                )
            )
        ]
    )

    fig.update_layout(get_plotly_layout(
        "Sentiment Distribution",
        height=380
    ))

    fig.update_layout(showlegend=True)

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )


# ---------------------------------------------------
# Sentiment Bar Chart
# ---------------------------------------------------

def create_sentiment_bar_chart(summary):
    """
    Creates interactive bar chart showing sentiment
    counts with gradient colors.
    """

    categories = ["Positive", "Neutral", "Negative"]

    values = [
        summary.get("positive", 0),
        summary.get("neutral", 0),
        summary.get("negative", 0)
    ]

    bar_colors = [
        COLORS["positive"],
        COLORS["neutral"],
        COLORS["negative"]
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker=dict(
                    color=bar_colors,
                    line=dict(color="white", width=1.5),
                    cornerradius=8
                ),
                text=values,
                textposition="outside",
                textfont=dict(size=14, color=COLORS["text"]),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Count: %{y}<extra></extra>"
                )
            )
        ]
    )

    layout = get_plotly_layout(
        "Sentiment Count Distribution",
        height=380
    )

    layout.update(
        xaxis=dict(
            showgrid=False,
            showline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            showline=False
        )
    )

    fig.update_layout(layout)

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )


# ---------------------------------------------------
# Compound Score Distribution
# ---------------------------------------------------

def create_compound_distribution(df):
    """
    Creates histogram of compound sentiment scores
    showing the distribution across all comments.
    """

    if df is None or df.empty or "compound" not in df.columns:
        return json.dumps({})

    fig = go.Figure(
        data=[
            go.Histogram(
                x=df["compound"],
                nbinsx=30,
                marker=dict(
                    color=COLORS["primary"],
                    line=dict(color="white", width=1),
                    cornerradius=4
                ),
                opacity=0.85,
                hovertemplate=(
                    "Score Range: %{x}<br>"
                    "Count: %{y}<extra></extra>"
                )
            )
        ]
    )

    # Add vertical lines for thresholds
    fig.add_vline(
        x=-0.05,
        line_dash="dash",
        line_color=COLORS["negative"],
        annotation_text="Negative threshold",
        annotation_position="top left"
    )

    fig.add_vline(
        x=0.05,
        line_dash="dash",
        line_color=COLORS["positive"],
        annotation_text="Positive threshold",
        annotation_position="top right"
    )

    layout = get_plotly_layout(
        "Compound Score Distribution",
        height=380
    )

    layout.update(
        xaxis=dict(
            title="Compound Score",
            showgrid=False,
            range=[-1.1, 1.1]
        ),
        yaxis=dict(
            title="Frequency",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)"
        )
    )

    fig.update_layout(layout)

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )


# ---------------------------------------------------
# Word Cloud
# ---------------------------------------------------

def create_wordcloud(df):
    """
    Creates a word cloud image from analyzed comments.
    Saves to static/generated/wordcloud.png.
    """

    if df is None or df.empty:
        _create_empty_image("wordcloud.png", "No comments available")
        return

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"

    if text_col not in df.columns:
        _create_empty_image("wordcloud.png", "No text data")
        return

    text = " ".join(
        df[text_col].dropna().astype(str)
    )

    if not text.strip():
        _create_empty_image("wordcloud.png", "No text available")
        return

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=150,
        contour_width=1,
        contour_color=COLORS["primary"],
        prefer_horizontal=0.7,
        min_font_size=10,
        max_font_size=120
    )

    image = wc.generate(text)

    plt.figure(figsize=(14, 7))
    plt.imshow(image, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        os.path.join(OUTPUT_DIR, "wordcloud.png"),
        bbox_inches="tight",
        dpi=150,
        facecolor="white"
    )

    plt.close()


# ---------------------------------------------------
# Top Words Charts
# ---------------------------------------------------

def top_words(df, sentiment, n=10):
    """Returns top n words for a given sentiment."""

    if df is None or df.empty:
        return [], []

    if "sentiment" not in df.columns:
        return [], []

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"

    if text_col not in df.columns:
        return [], []

    comments = df[
        df["sentiment"] == sentiment
    ][text_col]

    comments = comments.dropna().astype(str)
    comments = comments[comments.str.strip() != ""]

    if len(comments) == 0:
        return [], []

    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=n
        )

        X = vectorizer.fit_transform(comments)

    except ValueError:
        return [], []

    words = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1

    # Sort by count descending
    sorted_indices = counts.argsort()[::-1]
    words = [words[i] for i in sorted_indices]
    counts = [int(counts[i]) for i in sorted_indices]

    return words, counts


def create_top_words_chart(df, sentiment, color):
    """
    Creates interactive horizontal bar chart for top words.
    Returns Plotly JSON.
    """

    words, counts = top_words(df, sentiment)

    if not words:
        return json.dumps({})

    # Reverse for horizontal bar chart (top word at top)
    words = words[::-1]
    counts = counts[::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=words,
                orientation="h",
                marker=dict(
                    color=color,
                    line=dict(color="white", width=1),
                    cornerradius=6
                ),
                text=counts,
                textposition="outside",
                textfont=dict(size=12),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Count: %{x}<extra></extra>"
                )
            )
        ]
    )

    layout = get_plotly_layout(
        f"Top {sentiment} Words",
        height=380
    )

    layout.update(
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            showline=False
        ),
        yaxis=dict(
            showgrid=False,
            showline=False
        )
    )

    fig.update_layout(layout)

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )


def create_positive_chart(df):
    """Creates Top Positive Words chart."""
    return create_top_words_chart(
        df, "Positive", COLORS["positive"]
    )


def create_negative_chart(df):
    """Creates Top Negative Words chart."""
    return create_top_words_chart(
        df, "Negative", COLORS["negative"]
    )


# ---------------------------------------------------
# Confusion Matrix Heatmap
# ---------------------------------------------------

def create_confusion_matrix_chart(ml_metrics, model_name="logistic_regression"):
    """
    Creates interactive confusion matrix heatmap.
    Returns Plotly JSON.
    """

    if not ml_metrics:
        return json.dumps({})

    model_data = ml_metrics.get(model_name, {})
    cm = model_data.get("confusion_matrix", [])
    labels = model_data.get("labels", [])

    if not cm or not labels:
        return json.dumps({})

    cm_array = np.array(cm)

    # Colorscale
    colorscale = [
        [0, "#f0f0ff"],
        [0.5, "#a78bfa"],
        [1, "#6366f1"]
    ]

    # Create text annotations
    text_annotations = []
    for row in cm_array:
        text_row = []
        for val in row:
            text_row.append(str(int(val)))
        text_annotations.append(text_row)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=cm_array,
                x=labels,
                y=labels,
                colorscale=colorscale,
                text=text_annotations,
                texttemplate="%{text}",
                textfont=dict(size=16, color="white"),
                hovertemplate=(
                    "Predicted: %{x}<br>"
                    "Actual: %{y}<br>"
                    "Count: %{z}<extra></extra>"
                ),
                showscale=False
            )
        ]
    )

    display_name = model_name.replace("_", " ").title()

    layout = get_plotly_layout(
        f"Confusion Matrix — {display_name}",
        height=380
    )

    layout.update(
        xaxis=dict(
            title="Predicted Label",
            showgrid=False
        ),
        yaxis=dict(
            title="Actual Label",
            showgrid=False,
            autorange="reversed"
        )
    )

    fig.update_layout(layout)

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )


# ---------------------------------------------------
# Model Comparison Chart
# ---------------------------------------------------

def create_model_comparison_chart(ml_metrics):
    """
    Creates grouped bar chart comparing LR and NB metrics.
    Returns Plotly JSON.
    """

    if not ml_metrics:
        return json.dumps({})

    lr = ml_metrics.get("logistic_regression", {})
    nb = ml_metrics.get("naive_bayes", {})

    if not lr and not nb:
        return json.dumps({})

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    lr_values = [
        lr.get("accuracy", 0),
        lr.get("precision", 0),
        lr.get("recall", 0),
        lr.get("f1_score", 0)
    ]

    nb_values = [
        nb.get("accuracy", 0),
        nb.get("precision", 0),
        nb.get("recall", 0),
        nb.get("f1_score", 0)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Logistic Regression",
                x=metrics,
                y=lr_values,
                marker=dict(
                    color=COLORS["primary"],
                    cornerradius=6
                ),
                text=[f"{v:.1f}%" for v in lr_values],
                textposition="outside",
                textfont=dict(size=12)
            ),
            go.Bar(
                name="Naive Bayes",
                x=metrics,
                y=nb_values,
                marker=dict(
                    color=COLORS["secondary"],
                    cornerradius=6
                ),
                text=[f"{v:.1f}%" for v in nb_values],
                textposition="outside",
                textfont=dict(size=12)
            )
        ]
    )

    layout = get_plotly_layout(
        "Model Performance Comparison",
        height=400
    )

    layout.update(
        barmode="group",
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            showline=False,
            range=[0, 110]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_layout(layout)

    return json.dumps(
        fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )


# ---------------------------------------------------
# Helper: Empty Image
# ---------------------------------------------------

def _create_empty_image(filename, message):
    """Creates a placeholder image with a message."""

    plt.figure(figsize=(12, 6))

    plt.text(
        0.5, 0.5, message,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=18,
        color="#94a3b8",
        fontfamily="Inter"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

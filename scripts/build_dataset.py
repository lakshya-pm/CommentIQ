"""
scripts/build_dataset.py
========================
Fetch YouTube comments and pre-label them with RoBERTa for human review.

Usage:
    python scripts/build_dataset.py --videos VIDEO_IDS_FILE --out data/raw_dataset.csv

VIDEO_IDS_FILE is a plain text file with one YouTube URL or ID per line.
The script fetches up to --limit comments per video, pre-labels each
with the RoBERTa model, and writes a CSV with columns:

    video_id | text | roberta_label | roberta_score | human_label

'human_label' is blank — open the CSV in Excel/Sheets, review each row,
and fill in Positive / Negative / Neutral where you disagree with RoBERTa.
Leave it blank where you agree; train.py treats blank as RoBERTa's label.

Target: 1500+ comments across ≥5 varied video topics.
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.youtube import fetch_video_from_url
from modules import transformer as _t

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default video list — diverse topics to avoid domain bias
DEFAULT_VIDEOS = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",   # music (mixed sentiment)
    "https://www.youtube.com/watch?v=9bZkp7q19f0",   # K-Pop (positive-heavy)
    "https://www.youtube.com/watch?v=JGwWNGJdvx8",   # music (positive)
    "https://www.youtube.com/watch?v=OPf0YbXqDm0",   # tech review (mixed)
    "https://www.youtube.com/watch?v=6Ejga4kJUts",   # educational (positive)
]


def fetch_and_label(video_url: str, limit: int) -> pd.DataFrame:
    """
    Fetch comments from one video and pre-label with RoBERTa.

    Returns a DataFrame with columns:
        video_id, text, roberta_label, roberta_score, human_label
    """
    logger.info("Fetching %d comments from: %s", limit, video_url)

    try:
        data = fetch_video_from_url(video_url, limit=limit)
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", video_url, exc)
        return pd.DataFrame()

    comments = data.get("comments_list", [])
    video_id = data.get("video_id", video_url)

    if not comments:
        logger.warning("No comments found for %s", video_url)
        return pd.DataFrame()

    logger.info("Got %d comments. Running RoBERTa…", len(comments))

    roberta_results = _t.transformer_predict(comments)

    rows = []
    for i, comment in enumerate(comments):
        if i < len(roberta_results):
            label = roberta_results[i]["label"]
            score = roberta_results[i]["score"]
        else:
            label, score = "", 0.0

        rows.append({
            "video_id":      video_id,
            "text":          comment,
            "roberta_label": label,
            "roberta_score": round(score, 4),
            "human_label":   "",   # ← Fill this in Excel/Sheets
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build in-domain labelling dataset")
    parser.add_argument(
        "--videos",
        help="Path to text file with one YouTube URL per line (default: built-in list)",
    )
    parser.add_argument(
        "--out",
        default="data/raw_dataset.csv",
        help="Output CSV path (default: data/raw_dataset.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Max comments per video (default: 300)",
    )
    args = parser.parse_args()

    # Load video URLs
    if args.videos and os.path.exists(args.videos):
        with open(args.videos) as f:
            video_urls = [line.strip() for line in f if line.strip()]
    else:
        logger.info("No --videos file given; using built-in default list (%d videos)", len(DEFAULT_VIDEOS))
        video_urls = DEFAULT_VIDEOS

    all_frames = []
    for url in video_urls:
        df = fetch_and_label(url, limit=args.limit)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.error("No data collected. Check your API key and internet connection.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)

    # Deduplicate exact duplicate comments across videos
    combined = combined.drop_duplicates(subset=["text"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    combined.to_csv(args.out, index=False, encoding="utf-8-sig")

    logger.info(
        "Dataset written to %s\n"
        "  Total comments : %d\n"
        "  Videos         : %d\n\n"
        "Next step: open %s in Excel/Sheets and fill in 'human_label'\n"
        "where you disagree with 'roberta_label', then run:\n"
        "  python scripts/train.py\n",
        args.out, len(combined), len(video_urls), args.out,
    )


if __name__ == "__main__":
    main()

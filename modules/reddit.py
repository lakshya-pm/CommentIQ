"""
===========================================================
Reddit Module
CommentIQ - AI-Powered Comment Intelligence
===========================================================

Features
--------
✔ Analyze Reddit post using URL
✔ Search Reddit posts using keyword
✔ Fetch post details
✔ Fetch comments
"""

import praw

from config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT
)


# ---------------------------------------------------
# Lazy Reddit API Client
# ---------------------------------------------------

_reddit_client = None

def get_reddit_client():
    """
    Returns (and lazily initializes) the Reddit PRAW client.
    Raises a clear error if credentials are missing.
    """
    global _reddit_client

    if _reddit_client is None:
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            raise Exception(
                "Reddit API credentials not found. Please add "
                "REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and "
                "REDDIT_USER_AGENT to your .env file."
            )
        _reddit_client = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT or "CommentIQ/1.0"
        )

    return _reddit_client


# ---------------------------------------------------
# Fetch Reddit Post Details
# ---------------------------------------------------

def fetch_post_details(submission):

    details = {

        "title": submission.title,

        "subreddit": submission.subreddit.display_name,

        "author": str(submission.author),

        "score": submission.score,

        "comments_count": submission.num_comments,

        "url": submission.url,

        "created": submission.created_utc

    }

    return details


# ---------------------------------------------------
# Fetch Comments
# ---------------------------------------------------

def fetch_comments(submission, limit=100):

    submission.comments.replace_more(limit=0)

    comments = []

    for comment in submission.comments.list():

        if hasattr(comment, "body"):

            if comment.body != "[deleted]" and len(comment.body.split()) > 3:

                comments.append(comment.body)

        if len(comments) >= limit:

            break

    return comments


# ---------------------------------------------------
# Analyze Reddit URL
# ---------------------------------------------------

def fetch_post_from_url(url, limit=500):
    """
    Fetch a Reddit post and its comments.

    Args:
        url   : Reddit post URL
        limit : Maximum number of top-level comments to fetch (default 500)
    """
    submission = get_reddit_client().submission(url=url)

    details = fetch_post_details(submission)

    # Pass the user-selected limit through (previously hardcoded to 100)
    details["comments_list"] = fetch_comments(submission, limit=limit)

    return details


# ---------------------------------------------------
# Search Reddit
# ---------------------------------------------------

def search_post(keyword):

    subreddit = get_reddit_client().subreddit("all")

    posts = subreddit.search(keyword, limit=10)

    for submission in posts:

        try:

            comments = fetch_comments(submission)

            if len(comments) > 0:

                details = fetch_post_details(submission)

                details["comments_list"] = comments

                return details

        except Exception:

            continue

    raise Exception("No suitable Reddit post found.")

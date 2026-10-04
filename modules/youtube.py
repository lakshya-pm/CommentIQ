"""
===========================================================
YouTube Module
CommentIQ - AI-Powered Comment Intelligence
===========================================================

This module contains all YouTube related functions.

Features:
---------
✔ Search videos by keyword
✔ Fetch comments using video URL
✔ Fetch video information
✔ Support normal YouTube videos
✔ Support YouTube Shorts
✔ Support youtu.be links
✔ Support embedded YouTube videos
✔ Keep short social-media comments
✔ Format published date neatly
✔ Return structured data

Author: Lakshya Marwaha
"""

import re
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import YOUTUBE_API_KEY


# ---------------------------------------------------
# Lazy YouTube API Client
# ---------------------------------------------------

_youtube_client = None

def get_youtube_client():
    """
    Returns (and lazily initializes) the YouTube API client.
    Raises a clear error if the API key is missing.
    """
    global _youtube_client

    if _youtube_client is None:
        if not YOUTUBE_API_KEY:
            raise Exception(
                "YouTube API key not found. Please add YOUTUBE_API_KEY to your .env file."
            )
        _youtube_client = build(
            "youtube",
            "v3",
            developerKey=YOUTUBE_API_KEY
        )

    return _youtube_client


# ---------------------------------------------------
# Extract Video ID
# ---------------------------------------------------

def extract_video_id(url):
    """
    Extracts the YouTube video ID from different
    YouTube URL formats.

    Supported formats:

    1. Normal YouTube video:
       https://www.youtube.com/watch?v=VIDEO_ID

    2. Short YouTube URL:
       https://youtu.be/VIDEO_ID

    3. Embedded YouTube video:
       https://www.youtube.com/embed/VIDEO_ID

    4. YouTube Shorts:
       https://www.youtube.com/shorts/VIDEO_ID

    Returns:
        Video ID string or None
    """

    if not url:
        return None

    url = url.strip()

    # ------------------------------------------------
    # Make sure URL has a scheme
    # ------------------------------------------------

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    try:
        parsed = urlparse(url)

    except Exception:
        return None

    hostname = parsed.netloc.lower()

    # Remove www. if present
    hostname = hostname.replace("www.", "")

    path = parsed.path.strip("/")

    # ------------------------------------------------
    # Format 1: youtube.com/watch?v=VIDEO_ID
    # ------------------------------------------------

    if hostname in (
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    ):

        query = parse_qs(parsed.query)

        video_ids = query.get("v")

        if video_ids:

            video_id = video_ids[0]

            if re.fullmatch(
                r"[a-zA-Z0-9_-]{11}",
                video_id
            ):
                return video_id

    # ------------------------------------------------
    # Format 2: youtu.be/VIDEO_ID
    # ------------------------------------------------

    if hostname == "youtu.be":

        video_id = path.split("/")[0]

        if re.fullmatch(
            r"[a-zA-Z0-9_-]{11}",
            video_id
        ):
            return video_id

    # ------------------------------------------------
    # Format 3: youtube.com/embed/VIDEO_ID
    # ------------------------------------------------

    if hostname in (
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    ):

        path_parts = path.split("/")

        if (
            len(path_parts) >= 2
            and path_parts[0].lower() == "embed"
        ):

            video_id = path_parts[1]

            if re.fullmatch(
                r"[a-zA-Z0-9_-]{11}",
                video_id
            ):
                return video_id

    # ------------------------------------------------
    # Format 4: youtube.com/shorts/VIDEO_ID
    # ------------------------------------------------

    if hostname in (
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    ):

        path_parts = path.split("/")

        if (
            len(path_parts) >= 2
            and path_parts[0].lower() == "shorts"
        ):

            video_id = path_parts[1]

            if re.fullmatch(
                r"[a-zA-Z0-9_-]{11}",
                video_id
            ):
                return video_id

    return None


# ---------------------------------------------------
# Format Published Date
# ---------------------------------------------------

def format_published_date(published_at):
    """
    Converts YouTube's ISO 8601 timestamp into
    a cleaner dashboard-friendly format.

    Example:

    Input:
        2026-08-13T17:29:59Z

    Output:
        13 Aug 2026 • 17:29:59 UTC
    """

    if not published_at:
        return "Unknown"

    try:

        # Convert Z to UTC offset
        formatted_date = datetime.fromisoformat(
            published_at.replace(
                "Z",
                "+00:00"
            )
        )

        return formatted_date.strftime(
            "%d %b %Y • %H:%M:%S UTC"
        )

    except (ValueError, TypeError):

        # If formatting fails, return original value
        return published_at


# ---------------------------------------------------
# Fetch Video Details
# ---------------------------------------------------

def fetch_video_details(video_id):
    """
    Fetches YouTube video information.

    Returns:

    - Video ID
    - Title
    - Channel
    - Published date
    - Thumbnail
    - Views
    - Likes
    - Comment count
    """

    if not video_id:
        return None

    response = get_youtube_client().videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    items = response.get(
        "items",
        []
    )

    if not items:
        return None

    video = items[0]

    snippet = video.get(
        "snippet",
        {}
    )

    stats = video.get(
        "statistics",
        {}
    )

    # ------------------------------------------------
    # Get best available thumbnail
    # ------------------------------------------------

    thumbnails = snippet.get(
        "thumbnails",
        {}
    )

    thumbnail = ""

    if "high" in thumbnails:

        thumbnail = thumbnails["high"].get(
            "url",
            ""
        )

    elif "medium" in thumbnails:

        thumbnail = thumbnails["medium"].get(
            "url",
            ""
        )

    elif "default" in thumbnails:

        thumbnail = thumbnails["default"].get(
            "url",
            ""
        )

    # ------------------------------------------------
    # Format published date
    # ------------------------------------------------

    published_at = snippet.get(
        "publishedAt"
    )

    published = format_published_date(
        published_at
    )

    # ------------------------------------------------
    # Build video details
    # ------------------------------------------------

    details = {

        "video_id": video_id,

        "title": snippet.get(
            "title",
            "Unknown"
        ),

        "channel": snippet.get(
            "channelTitle",
            "Unknown"
        ),

        "published": published,

        "thumbnail": thumbnail,

        "views": int(
            stats.get(
                "viewCount",
                0
            )
        ),

        "likes": int(
            stats.get(
                "likeCount",
                0
            )
        ),

        "comments": int(
            stats.get(
                "commentCount",
                0
            )
        )

    }

    return details


# ---------------------------------------------------
# Fetch Comments
# ---------------------------------------------------

def fetch_comments(video_id, limit=500):
    """
    Fetches top-level comments from YouTube.

    Works for:

    - Normal YouTube videos
    - YouTube Shorts

    Short comments are also kept.

    Examples:

    - ❤️
    - Lovely ❤️
    - Great!
    - Wow!
    - Amazing video
    """

    comments = []

    next_page = None

    while len(comments) < limit:

        try:

            response = get_youtube_client().commentThreads().list(

                part="snippet",

                videoId=video_id,

                maxResults=100,

                textFormat="plainText",

                pageToken=next_page

            ).execute()

        except HttpError:

            # Comments may be disabled or unavailable
            break

        # ------------------------------------------------
        # Process comments
        # ------------------------------------------------

        for item in response.get(
            "items",
            []
        ):

            try:

                comment = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["snippet"]
                    ["textDisplay"]
                )

            except KeyError:

                continue

            # ------------------------------------------------
            # Keep every non-empty comment
            # ------------------------------------------------

            if comment and comment.strip():

                comments.append(
                    comment.strip()
                )

            if len(comments) >= limit:

                break

        # ------------------------------------------------
        # Get next page
        # ------------------------------------------------

        next_page = response.get(
            "nextPageToken"
        )

        if not next_page:

            break

    return comments[:limit]


# ---------------------------------------------------
# Analyze Video From URL
# ---------------------------------------------------

def fetch_video_from_url(url, limit=500):
    """
    Complete YouTube fetch.

    Supports:

    - Normal YouTube videos
    - youtu.be links
    - Embedded videos
    - YouTube Shorts

    Returns:

    {
        video_id,
        title,
        channel,
        published,
        thumbnail,
        views,
        likes,
        comments,
        comments_list
    }
    """

    # ------------------------------------------------
    # Extract Video ID
    # ------------------------------------------------

    video_id = extract_video_id(
        url
    )

    if video_id is None:

        raise Exception(
            "Invalid YouTube URL"
        )

    # ------------------------------------------------
    # Fetch video details
    # ------------------------------------------------

    details = fetch_video_details(
        video_id
    )

    if details is None:

        raise Exception(
            "YouTube video not found"
        )

    # ------------------------------------------------
    # Fetch comments
    # ------------------------------------------------

    comments = fetch_comments(
        video_id,
        limit=limit
    )

    # ------------------------------------------------
    # Add comments to details
    # ------------------------------------------------

    details[
        "comments_list"
    ] = comments

    return details


# ---------------------------------------------------
# Search Video By Keyword
# ---------------------------------------------------

def search_video(keyword):
    """
    Searches for suitable YouTube videos.

    Returns the first video that has comments.
    """

    search = get_youtube_client().search().list(

        q=keyword,

        part="snippet",

        type="video",

        maxResults=5

    ).execute()

    items = search.get(
        "items",
        []
    )

    if not items:

        raise Exception(
            "No videos found."
        )

    # ------------------------------------------------
    # Check each search result
    # ------------------------------------------------

    for item in items:

        try:

            video_id = (
                item["id"]["videoId"]
            )

        except KeyError:

            continue

        try:

            details = fetch_video_details(
                video_id
            )

            if details is None:

                continue

            comments = fetch_comments(
                video_id
            )

            if len(comments) > 0:

                details[
                    "comments_list"
                ] = comments

                return details

        except HttpError:

            continue

    raise Exception(
        "No videos with comments found."
    )

"""
tests/test_routes.py
====================
Smoke tests for Flask routes using mocked YouTube/Reddit API calls.

We mock the external data fetching so these tests:
  - Work without any API keys
  - Are fast (no network I/O)
  - Test the full request → response cycle
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as flask_app


SAMPLE_YOUTUBE_DATA = {
    "title":         "Test Video",
    "channel":       "Test Channel",
    "views":         100000,
    "likes":         5000,
    "published":     "2024-01-01",
    "thumbnail":     "https://example.com/thumb.jpg",
    "comments_list": [
        "This video is absolutely amazing I love it",
        "Terrible waste of time I hate this",
        "The video covers the topic in a straightforward way",
        "Great content keep up the good work really",
        "I will never watch this channel again bad",
    ],
}

SAMPLE_REDDIT_DATA = {
    "title":         "Test Reddit Post",
    "subreddit":     "test",
    "author":        "testuser",
    "score":         1000,
    "comments_list": [
        "This is really helpful thanks",
        "Totally disagree this is wrong",
        "Interesting perspective here",
    ],
}


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app.test_client() as c:
        yield c


# ---------------------------------------------------
# Home page
# ---------------------------------------------------

class TestHomeRoute:

    def test_home_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_home_contains_title(self, client):
        response = client.get("/")
        assert b"CommentIQ" in response.data


# ---------------------------------------------------
# /analyze — YouTube
# ---------------------------------------------------

class TestAnalyzeYouTube:

    @patch("app.fetch_video_from_url", return_value=SAMPLE_YOUTUBE_DATA)
    def test_youtube_url_returns_200(self, mock_fetch, client):
        response = client.post("/analyze", data={
            "url": "https://www.youtube.com/watch?v=test123",
            "comment_limit": "100",
        })
        assert response.status_code == 200
        mock_fetch.assert_called_once()

    @patch("app.fetch_video_from_url", return_value=SAMPLE_YOUTUBE_DATA)
    def test_youtu_be_short_url_works(self, mock_fetch, client):
        response = client.post("/analyze", data={
            "url": "https://youtu.be/test123",
            "comment_limit": "100",
        })
        assert response.status_code == 200

    @patch("app.fetch_video_from_url", return_value=SAMPLE_YOUTUBE_DATA)
    def test_dashboard_contains_sentiment_label(self, mock_fetch, client):
        response = client.post("/analyze", data={
            "url": "https://youtu.be/test123",
            "comment_limit": "100",
        })
        # Dashboard should mention at least one sentiment label
        body = response.data
        assert b"Positive" in body or b"Negative" in body or b"Neutral" in body


# ---------------------------------------------------
# /analyze — Reddit
# ---------------------------------------------------

class TestAnalyzeReddit:

    @patch("app.fetch_post_from_url", return_value=SAMPLE_REDDIT_DATA)
    def test_reddit_url_returns_200(self, mock_fetch, client):
        response = client.post("/analyze", data={
            "url": "https://www.reddit.com/r/test/comments/abc123/test_post/",
            "comment_limit": "100",
        })
        assert response.status_code == 200
        mock_fetch.assert_called_once()

    @patch("app.fetch_post_from_url", return_value=SAMPLE_REDDIT_DATA)
    def test_reddit_passes_comment_limit(self, mock_fetch, client):
        """comment_limit must be forwarded to fetch_post_from_url."""
        client.post("/analyze", data={
            "url": "https://www.reddit.com/r/test/comments/abc123/test/",
            "comment_limit": "200",
        })
        call_kwargs = mock_fetch.call_args
        # The limit kwarg should be 200
        assert call_kwargs.kwargs.get("limit") == 200


# ---------------------------------------------------
# /analyze — Input validation
# ---------------------------------------------------

class TestAnalyzeInputValidation:

    def test_empty_url_returns_error(self, client):
        response = client.post("/analyze", data={"url": ""})
        assert response.status_code == 200
        assert b"Please enter" in response.data or b"valid URL" in response.data

    def test_unsupported_url_returns_error(self, client):
        response = client.post("/analyze", data={
            "url": "https://twitter.com/user/status/123",
            "comment_limit": "100",
        })
        assert response.status_code == 200
        assert b"Unsupported" in response.data

    @patch("app.fetch_video_from_url",
           side_effect=Exception("quotaExceeded"))
    def test_quota_exceeded_shows_friendly_message(self, mock_fetch, client):
        response = client.post("/analyze", data={
            "url": "https://youtu.be/test123",
            "comment_limit": "100",
        })
        assert response.status_code == 200
        assert b"quota" in response.data.lower()


# ---------------------------------------------------
# /download
# ---------------------------------------------------

class TestDownloadRoute:

    def test_download_invalid_id_returns_400(self, client):
        response = client.get("/download?id=../../../etc/passwd")
        assert response.status_code == 400

    def test_download_unknown_id_returns_404(self, client):
        response = client.get("/download?id=00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    def test_download_no_id_returns_400(self, client):
        response = client.get("/download")
        assert response.status_code == 400

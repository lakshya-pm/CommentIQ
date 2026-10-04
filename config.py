# ==========================================
# CONFIGURATION FILE
# CommentIQ - AI-Powered Comment Intelligence
# ==========================================

import os
from dotenv import load_dotenv

# Load .env from the same directory as this file (always works regardless of CWD)
_base_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_base_dir, ".env")
load_dotenv(dotenv_path=_env_path, override=True)

# ----------------------------
# YouTube API Configuration
# ----------------------------

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ----------------------------
# Reddit API Configuration
# ----------------------------

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

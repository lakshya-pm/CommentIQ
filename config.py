# ==========================================
# CONFIGURATION FILE
# CommentIQ - NLP Sentiment Analysis Project
# ==========================================

import os
from dotenv import load_dotenv

# Load .env from project root regardless of working directory
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

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT")

# ----------------------------
# Optional feature flags
# ----------------------------

# Set USE_LORA_ADAPTER=true in .env to use the LoRA-fine-tuned RoBERTa adapter
# instead of the base cardiffnlp model.  Requires running scripts/finetune_lora.py first.
USE_LORA_ADAPTER = os.getenv("USE_LORA_ADAPTER", "false").lower() == "true"

# Set DEMO_MODE=true to load a bundled sample result without API keys.
# Useful for live demos where API quota is a concern.
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

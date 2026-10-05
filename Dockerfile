FROM python:3.11-slim

# Keeps Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures stdout/stderr are flushed (important for logging in Docker)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for numpy / matplotlib / wordcloud
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time so first request is fast
RUN python -c "\
import nltk; \
nltk.download('vader_lexicon', quiet=True); \
nltk.download('punkt',         quiet=True); \
nltk.download('punkt_tab',     quiet=True); \
nltk.download('stopwords',     quiet=True); \
nltk.download('wordnet',       quiet=True); \
nltk.download('omw-1.4',       quiet=True); \
"

# Copy application source
COPY . .

# Create runtime directories
RUN mkdir -p downloads/models downloads/results static/generated reports data

# Expose Flask port
EXPOSE 5000

# Run with gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]

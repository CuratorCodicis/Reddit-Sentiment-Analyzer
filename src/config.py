"""
Configuration Module for Reddit Sentiment & Trend Analyzer

This module centralizes all configuration settings for the application:
- Environment variables
- API settings
- Database settings
- Analysis parameters
- Visualization settings
- LLM settings
"""

import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

# Application version
APP_VERSION = "0.1.0"

# Reddit API Settings
REDDIT_API = {
    "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
    "user_agent": os.getenv(
        "REDDIT_USER_AGENT",
        f"Python:RedditSentimentAnalyzer:{APP_VERSION} (by /u/Senti-Analyzer)",
    ),
}

# Detect if running inside Docker (Docker Compose sets this automatically)
DOCKERIZED = os.getenv("DOCKERIZED", "false").lower() == "true"

# MongoDB settings
if DOCKERIZED:
    logging.info("🛠 Running inside Docker - Using internal MongoDB service")
    MONGODB = {
        "uri": "mongodb://mongo:27017",  # Use service name "mongo" from docker-compose
        "db_name": "reddit_sentiment",
        "collections": {
            "posts": "posts",
            "comments": "comments"
        }
    }
else:
    logging.info("Running locally - Using MongoDB settings from .env")
    MONGODB = {
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("MONGO_DB_NAME", "reddit_sentiment"),
        "collections": {
            "posts": "posts",
            "comments": "comments"
        }
    }

# Rate limiting settings
RATE_LIMIT = {
    "request_delay": int(
        os.getenv("REQUEST_DELAY", "5")
    ),  # seconds between API requests
}

# Sentiment analysis settings
SENTIMENT_ANALYSIS = {
    "positive_threshold": 0.05,  # minimum score to be considered positive
    "negative_threshold": -0.05,  # maximum score to be considered negative
    "title_weight": 0.6,  # weight given to title sentiment (vs. selftext)
    "vader_weight": 0.5,  # weight given to VADER (vs. TextBlob)
    "min_text_length": 5,  # minimum text length to analyze
}

# Data preprocessing settings
PREPROCESSING = {
    "remove_emojis": True,
    "remove_special_chars": True,
    "min_words_for_spam": 3,  # minimum words to not be considered spam
    "reddit_specific_stopwords": [
        "reddit",
        "upvote",
        "downvote",
        "karma",
        "subreddit",
        "edit",
        "thread",
        "post",
        "comment",
        "deleted",
        "removed",
    ],
}

# Default visualization settings
VISUALIZATION = {
    "color_palette": {
        "positive": "#4CAF50",  # Green
        "neutral": "#9E9E9E",  # Gray
        "negative": "#F44336",  # Red
        "default": "#2196F3",  # Blue
    },
    "wordcloud": {
        "max_words": 100,
        "width": 800,
        "height": 400,
        "background_color": "white",
        "colormap": "viridis",
    },
    "chart_defaults": {
        "figsize_large": (12, 8),
        "figsize_medium": (10, 6),
        "figsize_small": (8, 4),
    },
}

# Default analysis settings
DEFAULT_SETTINGS = {
    "post_limit": 25,
    "comment_limit": 5,
    "use_cached": True,
    "remove_spam": True,
    "time_interval": "day",
}

# Sample subreddits for example
SAMPLE_SUBREDDITS = [
    "python",
    "technology",
    "programming",
    "datascience",
    "news",
    "politics",
    "gaming",
]

# LLM settings
LLM = {
    "enabled": os.getenv("LLM_ENABLED", "true").lower() == "true",          # Whether LLM features are enabled system-wide
    "model_path": os.getenv("LLM_MODEL_PATH", "models/phi-2.Q4_K_M.gguf"),  # Path to the model file (for Docker deployment, this should be in a volume-mapped directory)
    "context_length": int(os.getenv("LLM_CONTEXT_LENGTH", "2048")),         # Context window size (in tokens)
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "512")),                  # Maximum tokens to generate in responses
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),              # Temperature controls randomness (0.0-1.0, higher = more random)
    "use_gpu": os.getenv("LLM_USE_GPU", "true").lower() == "true",          # Whether to use GPU acceleration (if available)
    "gpu_layers": int(os.getenv("LLM_GPU_LAYERS", "32")),                   # Number of GPU layers to offload (for mixed GPU/CPU processing), -1 means use all available GPU layers
    "timeout": int(os.getenv("LLM_TIMEOUT", "30")),                         # Timeout in seconds for LLM operations
    "memory": {                                                             # Memory settings (in MB) for optimizing LLM performance, only relevant for some models/configurations
        "cpu": int(os.getenv("LLM_CPU_MEMORY", "4000")),  # CPU RAM allocation
        "gpu": int(os.getenv("LLM_GPU_MEMORY", "4000")),  # GPU VRAM allocation
    },
}


def get_config() -> Dict[str, Any]:
    """
    Get the full configuration as a dictionary.

    This function collects all configuration settings from the module
    and returns them as a single dictionary for easy access.

    Returns:
        Dict[str, Any]: Complete configuration with all settings

    Example:
        >>> config = get_config()
        >>> print(config["app_version"])
        "1.0.0"
    """
    # Create a comprehensive dictionary with all config settings
    # This makes it easy to access all settings in one place
    return {
        "app_version": APP_VERSION,  # Application version number
        "reddit_api": REDDIT_API,  # Reddit API credentials and settings
        "mongodb": MONGODB,  # MongoDB connection settings
        "rate_limit": RATE_LIMIT,  # API rate limiting parameters
        "sentiment_analysis": SENTIMENT_ANALYSIS,  # Sentiment analysis parameters
        "preprocessing": PREPROCESSING,  # Text preprocessing settings
        "visualization": VISUALIZATION,  # Visualization styling options
        "default_settings": DEFAULT_SETTINGS,  # Default user settings
        "sample_subreddits": SAMPLE_SUBREDDITS,  # Example subreddits for users
    }


def validate_config() -> bool:
    """
    Validate that essential configuration values are set.

    This function checks whether the required configuration settings
    (particularly API credentials) are properly set in the environment
    variables. It helps prevent runtime errors due to missing credentials.

    Returns:
        bool: True if configuration is valid, False otherwise

    Example:
        >>> if not validate_config():
        ...     print("Please set up your .env file before running")
        ...     exit(1)
    """
    # Check Reddit API settings - we need both client_id and client_secret
    # to authenticate with Reddit's API
    if not REDDIT_API["client_id"] or not REDDIT_API["client_secret"]:
        logging.error("Reddit API credentials not found in .env file")
        return False

    # Check MongoDB settings - we need a valid URI to connect to the database
    if not MONGODB["uri"]:
        logging.error("MongoDB URI not found in .env file")
        return False

    # If all checks pass, configuration is valid
    return True


# When imported as a module, validate the configuration
if __name__ != "__main__":
    if not validate_config():
        logging.warning(
            "Missing configuration values. Please check your .env file. "
            "The application may not function correctly."
        )

# Example usage
if __name__ == "__main__":
    # Print the current configuration
    import json

    # Validate configuration
    is_valid = validate_config()

    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration is invalid")

    # Print configuration (redacting sensitive values)
    config = get_config()

    # Redact sensitive values
    if config["reddit_api"]["client_id"]:
        config["reddit_api"]["client_id"] = "****"
    if config["reddit_api"]["client_secret"]:
        config["reddit_api"]["client_secret"] = "****"

    print(json.dumps(config, indent=2))

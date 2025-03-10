"""
Sentiment Analysis Module for Reddit Sentiment Analyzer

This module analyzes the sentiment of Reddit posts and comments using multiple NLP methods:
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- TextBlob

The module compares results from both methods to provide more robust sentiment analysis.
"""

import logging  # For logging events and errors
import nltk  # Natural Language Toolkit for NLP operations
from typing import List, Dict, Any, Tuple, Union  # Type hints
from textblob import TextBlob  # Simple library for text processing
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER sentiment analyzer

from config import SENTIMENT_ANALYSIS

# Configure logging to show the severity level and message
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Check if VADER lexicon is downloaded and download if needed
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
    logging.info("VADER lexicon is already downloaded.")
except LookupError:
    logging.info("Downloading VADER lexicon...")
    nltk.download("vader_lexicon")


class SentimentAnalyzer:
    """Class for sentiment analysis"""

    def __init__(self):
        """Initialize sentiment analyzers."""
        self.vader = SentimentIntensityAnalyzer()

    def analyze_text_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment using VADER.

        Args:
            text (str): The text to analyze

        Returns:
            Dict[str, float]: Dictionary with VADER sentiment scores
                - 'neg': Negative score (0-1)
                - 'neu': Neutral score (0-1)
                - 'pos': Positive score (0-1)
                - 'compound': Compound score (-1 to 1)
        """
        if not text or not isinstance(text, str):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        return self.vader.polarity_scores(text)

    def analyze_text_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment using TextBlob.

        Args:
            text (str): The text to analyze

        Returns:
            Dict[str, float]: Dictionary with TextBlob sentiment scores
                - 'polarity': Sentiment from -1 (negative) to 1 (positive)
                - 'subjectivity': Subjectivity from 0 (objective) to 1 (subjective)
        """
        if not text or not isinstance(text, str):
            return {"polarity": 0.0, "subjectivity": 0.0}

        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    def get_combined_sentiment(
        self, vader_score: float, textblob_score: float
    ) -> Tuple[str, float]:
        """
        Get combined sentiment label and score from VADER and TextBlob results.

        Args:
            vader_score (float): VADER compound score (-1 to 1)
            textblob_score (float): TextBlob polarity score (-1 to 1)

        Returns:
            Tuple[str, float]: (sentiment_label, combined_score)
                - sentiment_label: 'positive', 'negative', or 'neutral'
                - combined_score: Average of VADER and TextBlob scores (-1 to 1)
        """
        # Calculate average score (giving equal weight to both methods)
        combined_score = (vader_score + textblob_score) / 2

        # Determine sentiment label based on combined score
        if combined_score >= SENTIMENT_ANALYSIS["positive_threshold"]:
            label = "positive"
        elif combined_score <= SENTIMENT_ANALYSIS["negative_threshold"]:
            label = "negative"
        else:
            label = "neutral"

        return label, combined_score

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using both VADER and TextBlob and return combined results.
        This is the main method that combines multiple sentiment analysis approaches
        for more robust results.

        Args:
            text (str): The text to analyze

        Returns:
            Dict[str, Any]: Dictionary with all sentiment analysis results
                - text_snippet: A preview of the analyzed text
                - sentiment: The overall sentiment label ('positive', 'negative', 'neutral')
                - score: The combined sentiment score from -1 (negative) to 1 (positive)
                - vader: Detailed VADER analysis results
                - textblob: Detailed TextBlob analysis results

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze_text("I love this product! Highly recommended.")
            >>> result['sentiment']
            'positive'
            >>> result['score']
            0.75  # Example value, actual score may vary
        """
        # Get VADER scores
        vader_scores = self.analyze_text_vader(text)

        # Get TextBlob scores
        textblob_scores = self.analyze_text_textblob(text)

        # Combine the scores
        sentiment_label, combined_score = self.get_combined_sentiment(
            vader_scores["compound"], textblob_scores["polarity"]
        )

        # Return comprehensive result dictionary with all information
        return {
            # Truncate long text for readability
            "text_snippet": text[:100] + "..." if len(text) > 100 else text,
            # Overall sentiment classification
            "sentiment": sentiment_label,  # 'positive', 'neutral', or 'negative'
            # Combined numerical score
            "score": combined_score,  # -1 (very negative) to 1(very positive)
            # Results from each individual analyzer
            "vader": vader_scores,
            "textblob": textblob_scores,
        }

    def analyze_reddit_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a Reddit post or comment.

        Args:
            item (Dict[str, Any]): Dictionary containing Reddit post or comment data

        Returns:
            Dict[str, Any]: Same dictionary with added sentiment analysis
        """
        # Weights for title and selfext sentiment
        TITLE_WEIGHT = 0.5
        SELFTEXT_WEIGHT = 0.5

        # Create copy to avoid modifying the original
        result = item.copy()

        # Dtermine if it is a post (has title and selftext keys) or comment (has body key)
        if "title" in item or "selftext" in item:
            # Post - analyze title and selftext separately
            title_sentiment = self.analyze_text(
                item.get("title", "")
            )  # Default empty string, if title is None

            # Only analyze selftext if it exists and is not empty
            if item.get("selftext"):
                selftext_sentiment = self.analyze_text(item.get("selftext", ""))

                # Combined sentiment (equal weight to title and selftext)
                combined_score = (
                    title_sentiment["score"] * TITLE_WEIGHT
                    + selftext_sentiment["score"] * SELFTEXT_WEIGHT
                )

                # Add all sentiment data to the result
                result["sentiment"] = {
                    "title": title_sentiment,
                    "selftext": selftext_sentiment,
                    "combined": {
                        "sentiment": self.get_combined_sentiment(
                            title_sentiment["vader"]["compound"] * TITLE_WEIGHT
                            + selftext_sentiment["vader"]["compound"] * SELFTEXT_WEIGHT,
                            title_sentiment["textblob"]["polarity"] * TITLE_WEIGHT
                            + selftext_sentiment["textblob"]["polarity"]
                            * SELFTEXT_WEIGHT,
                        )[
                            0
                        ],  # get_combined_sentiment returns a tuple with the label at [0]
                        "score": combined_score,
                    },
                }
            else:
                # No selftext, just use title sentiment
                result["sentiment"] = {
                    "title": title_sentiment,
                    "combined": {
                        "sentiment": title_sentiment["sentiment"],
                        "score": title_sentiment["score"],
                    },
                }

        elif "body" in item:
            # Comment - analyze body
            body_sentiment = self.analyze_text(item.get("body", ""))
            result["sentiment"] = {
                "body": body_sentiment,
                "combined": {
                    "sentiment": body_sentiment["sentiment"],
                    "score": body_sentiment["score"],
                },
            }

        return result

    def analyze_reddit_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of a list of Reddit posts or comments.

        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing Reddit data

        Returns:
            List[Dict[str, Any]]: Same list with added sentiment analysis
        """
        results = []

        for item in data:
            analyzed_item = self.analyze_reddit_item(item)
            results.append(analyzed_item)

        logging.info(f"Analyzed sentiment of {len(results)} items.")
        return results


# Example usage
if __name__ == "__main__":
    # Example data
    sample_data = [
        {
            "id": "post1",
            "title": "I absolutely love the new features in Python 3.11!",
            "selftext": "The performance improvements are incredible. This is the best update ever.",
            "author": "happy_user",
        },
        {
            "id": "post2",
            "title": "Frustrated with the new API changes",
            "selftext": "The documentation is terrible and nothing works properly. Waste of time.",
            "author": "angry_dev",
        },
        {
            "id": "comment1",
            "body": "This is a neutral comment that doesn't express strong emotions.",
            "author": "neutral_user",
        },
    ]

    # Initialize the sentiment analyzer
    analyzer = SentimentAnalyzer()

    # Analyze the sample data
    analyzed_data = analyzer.analyze_reddit_data(sample_data)

    # Print the results
    for item in analyzed_data:
        if "title" in item:
            print(f"Post: {item['title']}")
            print(f"Sentiment: {item['sentiment']['combined']['sentiment']}")
            print(f"Score: {item['sentiment']['combined']['score']:.2f}")
        else:
            print(f"Comment: {item['body']}")
            print(f"Sentiment: {item['sentiment']['combined']['sentiment']}")
            print(f"Score: {item['sentiment']['combined']['score']:.2f}")
        print("---")

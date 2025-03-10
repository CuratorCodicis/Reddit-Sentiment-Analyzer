"""
Data Preprocessing Module for Reddit Sentiment Analyzer

This module handles cleaning and preprocessing of Reddit data before sentiment analysis.
- Clean text data (remove URLs, special characters, etc.)
- Remove spam and bot comments
- Filter posts and comments by keywords
- Normalize text for sentiment analysis
"""

import re  # Regular expressions for pattern matching
import string  # String constants and operations
import logging  # logging messages and errors
from typing import List, Dict, Any, Union  # Type hints for better documentation

# Set logging format
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clean_text(text: str) -> str:
    """
    Clean text data for better sentiment analysis by:
    - Converting to lowercase
    - Removing Reddit-specific formatting
    - Removing special characters
    - Removing extra whitespace

    - TODO: Removing URLs - not sure if wanted behaviour

    Args:
        text (str): The raw text to clean

    Returns:
        str: Cleaned text

    Example:
        >>> clean_text("Check out http://example.com! **IMPORTANT** [removed]")
        "check out important"
    """
    # Return empty string if text is None or not a string
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase for consistency
    text = text.lower()

    # Use regular expressions to remove/replace unwanted chars/patterns in text
    # -------------------------------------------------
    # re.sub() is a function that replaces patterns in text:
    # 1st parameter: pattern to search for
    # 2nd parameter: what to replace it with ('' means remove)
    # 3rd parameter: the text to search within
    #
    #   - r' ... ' means a raw string (treats backslashes literally)
    #   - \S+ means "one or more non-whitespace characters"
    #   - | means "OR" (matches any of these patterns)

    # Remove Reddit formatting (like [removed] or [deleted])
    # The pattern matches the exact string [removed] or [deleted]
    text = re.sub(r"\[removed\]|\[deleted\]", "", text)

    # Remove Reddit markdown formatting
    # - ** for bold text
    # - * for italic text
    # - ~~ for strikethrough
    # - __ for underline
    # - > for blockquotes
    text = re.sub(r"\*\*|\*|~~|__|>", "", text)

    # Remove special characters, keeping only alphanumeric and basic punctuation
    # The pattern [^\w\s.,!?] means:
    #   - ^ inside [] means excludes the following characters from the function (i.e. they will not be subbed)
    #   - \w means any word character (letters, numbers, underscore)
    #   - \s means any whitespace
    #   - .,!? are the specific punctuation to keep
    text = re.sub(r"[^\w\s.,!?]", "", text)

    # Remove extra whitespace (multiple whitespaces, tabs, newlines)
    # \s+ means "one or more whitespace characters"
    # Replace with a single space, then strip() to remove leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_by_keywords(
    data: List[Dict[str, Any]],
    keywords: List[str],
    text_fields: List[str] = ["title", "selftext", "body"],
) -> List[Dict[str, Any]]:
    """
    Filter posts or comments by specified keywords.

    Args:
        data (List[Dict]): List of post/comment dictionaries
        keywords (List[str]): List of keywords to filter by
        text_fields (List[str]): Fields in the dictionary to search for keywords

    Returns:
        List[Dict]: Filtered list of dictionaries that contain at least one keyword
    """
    if not keywords:
        return data

    # Convert keywords to lowercase for case-insensitive matching
    keywords = [keyword.lower() for keyword in keywords]

    filtered_data = []
    for item in data:
        # Check each text field for keyword matches
        for field in text_fields:
            if field in item and item[field]:
                text = item[field].lower()
                # If any keyword is found in the text, add the item and break the loop
                if any(keyword in text for keyword in keywords):
                    filtered_data.append(item)
                    break

    logging.info(
        f"Filtered data from {len(data)} items to {len(filtered_data)} items based on keywords."
    )
    return filtered_data


def identify_spam(
    data: List[Dict[str, Any]], threshold: int = 3
) -> List[Dict[str, Any]]:
    """
    Identify and filter out potential spam posts/comments based on simple heuristics.

    Args:
        data (List[Dict]): List of post/comment dictionaries
        threshold (int): Threshold for spam detection rules

    Returns:
        List[Dict]: Filtered list with spam removed
    """
    filtered_data = []

    for item in data:
        is_spam = False

        # Check for very short content that is probably not meaningful
        text_field = "body" if "body" in item else "selftext"
        if text_field in item and item[text_field]:
            text = item[text_field]

            # 1. Too short content (less than threshold words)
            if len(text.split()) < threshold:
                is_spam = True

            # 2. Repetitive characters (like "aaaaa")
            elif any(char * threshold in text for char in string.ascii_letters):
                is_spam = True

            # 3. All caps (SHOUTING AT ME)
            elif text.isupper():  # and len(text) > threshold * 2?
                is_spam = True
        # Only include non-spam items
        if not is_spam:
            filtered_data.append(item)

    spam_count = len(data) - len(filtered_data)
    if spam_count > 0:
        logging.info(f"Removed {spam_count} items identified as potential spam.")

    return filtered_data


def preprocess_data(
    data: List[Dict[str, Any]],
    keywords: List[str] = None,
    clean_fields: List[str] = [
        "title",
        "selftext",
        "body",
    ],  # 'selftext' for posts, 'body' for comments
    remove_spam: bool = True,
) -> List[Dict[str, Any]]:
    """
    Main preprocessing function that applies all preprocessing steps.

    Args:
        data (List[Dict]): List of post/comment dictionaries
        keywords (List[str], optional): Keywords to filter by
        clean_fields (List[str]): Fields to apply text cleaning to
        remove_spam (bool): Whether to remove potential spam

    Returns:
        List[Dict]: Preprocessed data
    """
    # Start with original data
    processed_data = data.copy()

    # DO NOT FILTER FOR KEYWORDS - only do that after loading from the database
    # Filter by keywords if provided
    # if keywords:
    #    processed_data = filter_by_keywords(processed_data, keywords)

    # Remove potential spam if requested
    if remove_spam:
        processed_data = identify_spam(processed_data)

    # Clean text fields
    for item in processed_data:
        for field in clean_fields:
            if field in item and item[field]:
                item[field + "_original"] = item[field]  # Save original text
                item[field] = clean_text(item[field])  # Replace with cleaned text

    return processed_data


# Example usage
if __name__ == "__main__":
    # Example data
    sample_data = [
        {
            "id": "abc123",
            "title": "Check out my new website! http://spam-site.com",
            "selftext": "**Please** visit my *awesome* website [here](http://spam-site.com)!",
            "score": 0,
        },
        {
            "id": "def456",
            "title": "Discussion about Python programming",
            "selftext": "I've been learning Python for a month now. What are some good resources?",
            "score": 42,
        },
    ]

    # Preprocess the sample data
    preprocessed = preprocess_data(sample_data, keywords=["python"])

    # Print the results
    for item in preprocessed:
        print(f"ID: {item['id']}")
        print(f"Title: {item['title']}")
        print(f"Text: {item['selftext']}")
        print("---")

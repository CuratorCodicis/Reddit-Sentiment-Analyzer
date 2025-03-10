"""
Visualization Module for Reddit Sentiment Analyzer

This module provides functions to create visualizations from analyzed Reddit data:
- Sentiment distribution (pie chart)
- Sentiment trends over time (line chart)
- Word clouds of frequent terms
- Interactive visualizations for the GUI
"""

import re  # Regular expressions
from collections import Counter  # For counting word occurrences efficiently
import logging  # For logging events and errors
import matplotlib.pyplot as plt  # Core plotting library
import numpy as np  # Numerical operations and arrays
from typing import List, Dict, Any, Optional, Tuple  # Type hints
from collections import Counter  # Efficient counting of item
from datetime import datetime  # Date and time handling
from wordcloud import WordCloud, STOPWORDS  # Word cloud generation
import pandas as pd  # Data analysis and manipulation
import seaborn as sns  # Statistical visualization
from matplotlib.figure import Figure  # For type hinting matplotlib figures
from io import BytesIO  # In-memory binary stream
import base64  # Converting binary data to text

from config import VISUALIZATION, SENTIMENT_ANALYSIS

# Configure logging to show severity level and message
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Set seaborn style for better-looking charts
sns.set_style("whitegrid")

# Add Reddit-specific terms to stopwords to remove them from text analysis
REDDIT_STOPWORDS = set(
    [
        "reddit",
        "upvote",
        "downvote",
        "comment",
        "post",
        "karma",
        "subreddit",
        "edit",
        "deleted",
        "removed",
        "submission",
        "thread",
        "would",
        "could",
        "also",
        "get",
        "one",
        "like",
        "even",
        "still",
    ]
)


def plot_sentiment_distribution(
    data: List[Dict[str, Any]],
    title: str = "Sentiment Distribution",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Create a pie chart showing the distribution of sentiments (positive, negative, neutral).

    Args:
        data (List[Dict]): List of dictionaries with sentiment analysis
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)

    Returns:
        Figure: Matplotlib figure object containing the visualization
    """
    # Count sentiment occurrences to keep track of the number of posts
    sentiment_counts = Counter()

    for item in data:
        # Check if the item has sentiment analysis results
        if "sentiment" in item and "combined" in item["sentiment"]:
            # Extract sentiment label
            sentiment = item["sentiment"]["combined"]["sentiment"]
            # Increment the count for this sentiment
            sentiment_counts[sentiment] += 1

    # If no sentiments found, return a message
    if sum(sentiment_counts.values()) == 0:
        # plt.subplot() returns a figure and its axes
        fig, ax = plt.subplots(figsize=figsize)
        # Add text at the position centered in the figure
        ax.text(
            0.5,
            0.5,
            "No sentiment data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        # Turn off the axes, no coordinate lines for the message
        ax.axis("off")
        return fig

    # Prepare data for the pie chart
    # Extract sentiment labels and their counts
    labels = sentiment_counts.keys()  # e.g. ['positive', 'neutral', 'negative']
    sizes = sentiment_counts.values()  # e.g. [10, 5, 3] meaning the counts

    # Set colors for each sentiment
    colors = VISUALIZATION["color_palette"]

    # Create a list of colors in the same order as the labels
    color_list = [colors.get(label, "#2196F3") for label in labels]

    # Create the pie chart
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create the pie chart on the axis
    # wedges: the pie slices
    # texts: the labels
    # autotexts: the percentage labels
    wedges, texts, autotexts = ax.pie(
        sizes,  # The values to plot
        labels=labels,  # Labels for each slice
        autopct="%1.1f%%",  # Format string for percentages (1 decimal place)
        colors=color_list,  # Colors for each slice
        startangle=90,  # Rotate so first slice starts at 90° i.e. the top
        shadow=False,  # No drop shadow effect
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis("equal")

    # Format the percentage labels for better readbility
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_weight("bold")

    # Add title to chart
    plt.title(title, fontsize=16, pad=20)  # pad adds space between title and plot

    # Add legend with count information
    # Create legend labels with counts
    legend_labels = [f"{label} ({count})" for label, count in zip(labels, sizes)]

    # Add the legend to the figure
    # bbox_to_anchor=(1, 0, 0.5, 1) positions the legend outside the pie chart to the right
    plt.legend(
        wedges,
        legend_labels,
        title="Sentiment",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    # Adjust layout so everything fits nicely
    plt.tight_layout()

    return fig


def plot_sentiment_over_time(
    data: List[Dict[str, Any]],
    time_interval: str = "day",
    title: str = "Sentiment Trend Over Time",
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Create a line chart showing sentiment trends over time.

    Args:
        data (List[Dict]): List of dictionaries with sentiment analysis
        time_interval (str): Time interval for grouping ('hour', 'day', 'week', 'month')
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)

    Returns:
        Figure: Matplotlib figure object containing the visualization
    """
    # Debug information
    logging.info(f"plot_sentiment_over_time: Processing {len(data)} items")

    # Count sentiment types in input data
    sentiment_types = [
        item["sentiment"]["combined"]["sentiment"]
        for item in data
        if "sentiment" in item and "combined" in item["sentiment"]
    ]
    sentiment_counts = {s: sentiment_types.count(s) for s in set(sentiment_types)}
    logging.info(f"Input data sentiment counts: {sentiment_counts}")

    # Convert data to pandas DataFrame for easier time-series handling
    df_data = []

    # Process each Reddit item
    for item in data:
        if (
            "sentiment" in item
            and "combined" in item["sentiment"]
            and "created_utc" in item
        ):
            try:
                # Convert Unix timestamp to datetime
                created_time = datetime.fromtimestamp(item["created_utc"])
                sentiment = item["sentiment"]["combined"]["sentiment"]
                score = item["sentiment"]["combined"]["score"]

                # Add to list of data points
                df_data.append(
                    {
                        "created_time": created_time,
                        "sentiment": sentiment,
                        "score": score,
                    }
                )
            except Exception as e:
                logging.warning(f"Skipping item due to timestamp conversion error: {e}")

    # Handle empty data case
    if not df_data:
        logging.warning("No valid data points for time series visualization")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No time-series data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Create DataFrame from collected data
    df = pd.DataFrame(df_data)
    logging.info(f"Created DataFrame with {len(df)} rows")
    # Debug info about DataFrame
    logging.info(
        f"DataFrame sentiment counts: {df['sentiment'].value_counts().to_dict()}"
    )

    # Set time as the index for time-series operations
    # This allows the use of pandas time-series functions
    df["created_time"] = pd.to_datetime(
        df["created_time"], errors="coerce"
    )  # DEBUG Ensures conversion without errors
    df.set_index("created_time", inplace=True)

    # Log time range
    logging.info(f"Time range: {df.index.min()} to {df.index.max()}")

    # Define resampling frequency based on time_interval
    # Determines how data is grouped over time (hourly, daily, etc.)
    resample_freq = {"hour": "H", "day": "D", "week": "W", "month": "M"}.get(
        time_interval.lower(), "D"
    )  # Default to day if invalid value
    logging.info(f"Using resampling frequency: {resample_freq}")

    # Check if DataFrame is empty after resampling
    if df.empty:
        logging.warning("DataFrame is empty after processing. Returning empty plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No valid data after resampling",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Calculate average sentiment score for each time interval
    # resamle() groups data by time periods and mean() calculates average score in each period
    sentiment_by_time = df["score"].resample(resample_freq).mean()
    logging.info(f"Resampled to {len(sentiment_by_time)} time points")

    # Drop days (or hours/weeks) that have no data:
    sentiment_by_time = sentiment_by_time.dropna()

    # Count sentiments for each time interval
    # Filter for each sentiment type
    # Resample to count occurrences in each time period
    positive_counts = df[df["sentiment"] == "positive"].resample(resample_freq).size()
    neutral_counts = df[df["sentiment"] == "neutral"].resample(resample_freq).size()
    negative_counts = df[df["sentiment"] == "negative"].resample(resample_freq).size()

    # Log counts after resampling
    logging.info(f"Positive counts after resampling: {positive_counts.to_dict()}")
    logging.info(f"Neutral counts after resampling: {neutral_counts.to_dict()}")
    logging.info(f"Negative counts after resampling: {negative_counts.to_dict()}")

    # Create figure with two subplots
    # The first subplot shows average score over time
    # The second subplot shows count of each sentiment type over time
    # sharex=True makes them share the same x-axis
    # height_ratios=[2, 1] makes the top plot twice as tall as the bottom plot
    fig, (ax1, ax2) = plt.subplots(
        2,  # 2 rows
        1,  # 1 column
        figsize=figsize,  # Figure size
        sharex=True,  # Share x-axis
        gridspec_kw={"height_ratios": [2, 1]},  # top plot is twice as tall
    )

    # Plot average sentiment score on the first subplot
    sentiment_by_time.plot(
        ax=ax1,  # Plot on the first subplot
        marker="o",  # Use circle markers at data points
        linestyle="-",  # Solid line
        linewidth=2,  # Line thickness
        markersize=6,  # Size of markers
        color="#2196F3",  # Blue color
    )

    # Add horizontal line at y=0 (neutral sentiment)
    # Visually separate positive from negative sentiment
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    # Color the background based on sentiment regions
    # This creates colored bands in the background to indicate sentiment regions
    ax1.axhspan(
        SENTIMENT_ANALYSIS["positive_threshold"], 1, facecolor="#E8F5E9", alpha=0.6
    )  # Light green for positive
    ax1.axhspan(
        SENTIMENT_ANALYSIS["negative_threshold"],
        SENTIMENT_ANALYSIS["positive_threshold"],
        facecolor="#EEEEEE",
        alpha=0.6,
    )  # Light gray for neutral
    ax1.axhspan(
        -1, SENTIMENT_ANALYSIS["negative_threshold"], facecolor="#FFEBEE", alpha=0.6
    )  # Light red for negative

    # Set y-axis limits and labels for the first subplot
    ax1.set_ylim(-1, 1)  # Sentiment scores range from -1 to 1
    ax1.set_ylabel("Average Sentiment Score", fontsize=12)
    ax1.set_title(title, fontsize=16)

    # For second subplot, ensure we're plotting all three sentiment lines
    # Plot sentiment counts on the second subplot
    # Fill with zeros for missing time periods
    all_times = sentiment_by_time.index

    # Initialize counts for all time periods
    positive_counts_all = pd.Series(0, index=all_times)
    neutral_counts_all = pd.Series(0, index=all_times)
    negative_counts_all = pd.Series(0, index=all_times)

    # Update with actual counts where available
    positive_counts_all.update(positive_counts)
    neutral_counts_all.update(neutral_counts)
    negative_counts_all.update(negative_counts)

    # Plot each sentiment type as a separate line
    pos_line = positive_counts_all.plot(
        ax=ax2, label="Positive", color="#4CAF50"
    )  # Green
    neu_line = neutral_counts_all.plot(ax=ax2, label="Neutral", color="#9E9E9E")  # Gray
    neg_line = negative_counts_all.plot(
        ax=ax2, label="Negative", color="#F44336"
    )  # Red

    # Verify that all lines were plotted
    logging.info(
        f"Plotted lines: Positive: {pos_line is not None}, "
        f"Neutral: {neu_line is not None}, Negative: {neg_line is not None}"
    )

    # Add labels and legend
    ax2.set_ylabel("Count", fontsize=12)
    ax2.legend(loc="upper left")

    # Set Y-ticks to be evenly spaced
    ax2.set_yticks(range(0, max(positive_counts_all.max(), neutral_counts_all.max(), negative_counts_all.max()) + 1, 5))

    # Set Y-axis limits
    ax2.set_ylim(0, max(positive_counts_all.max(), neutral_counts_all.max(), negative_counts_all.max()) * 1.1)

    # Add grid for better readability
    ax2.grid(True, linestyle="--", alpha=0.5)


    # Format x-axis dates based on the time interval
    # This makes the date format appropriate for the selected time interval
    if time_interval.lower() == "hour":
        # For hourly data, show month-day hour:minute
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d %H:%M"))
    elif time_interval.lower() == "day":
        # For daily data, show month-day
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
    else:
        # For weekly or monthly data, show year-month-day
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))

    if len(df.index) > 0:
        ax1.set_xlim(df.index.min(), df.index.max())


    # Adjust layout
    plt.tight_layout()
    # Reduce space between the two subplots
    plt.subplots_adjust(hspace=0.1)

    return fig


def generate_wordcloud(
    data: List[Dict[str, Any]],
    sentiment_filter: Optional[str] = None,
    max_words: int = 400,
    figsize: Tuple[int, int] = (10, 6),
    width: int = 800,
    height: int = 400,
) -> Figure:
    """
    Generate a word cloud from text in Reddit data.

    A word cloud is a visual representation of text data where the size of each word
    indicates its frequency or importance. This function creates word clouds from
    Reddit posts/comments, optionally filtered by sentiment.

    Args:
        data (List[Dict]): List of dictionaries with Reddit data
        sentiment_filter (str, optional): Filter by sentiment ('positive', 'negative', 'neutral')
        max_words (int): Maximum number of words to include
        figsize (Tuple[int, int]): Figure size (width, height) in inches
        width (int): Width of wordcloud image in pixels
        height (int): Height of wordcloud image in pixels

    Returns:
        Figure: Matplotlib figure object containing the word cloud
    """
    # Prepare stopwords (words to exclude from the word cloud)
    # Combine standard stopwords with Reddit-specific ones defined earlier
    all_stopwords = set(STOPWORDS).union(REDDIT_STOPWORDS)

    # Collect relevant text based on sentiment filter
    all_text = []

    # Process each Reddit item
    for item in data:
        # Skip items that do not match the sentiment filter (if specified)
        if sentiment_filter and "sentiment" in item and "combined" in item["sentiment"]:
            if item["sentiment"]["combined"]["sentiment"] != sentiment_filter:
                # Only allow items which label matches the filter specified
                continue

        # Handle posts - 'title' and 'body'
        if "title" in item:
            # Add the title text
            all_text.append(item["title"])

            # Add the post content if it exists
            if "selftext" in item and item["selftext"]:
                all_text.append(item["selftext"])

        # Handle comments - 'body'
        elif "body" in item and item["body"]:
            all_text.append(item["body"])

    # Combine all text into a single string
    # join() concatenates all strings in all_text with a space between them
    text = " ".join(all_text)

    # Handle the case where no text is available
    if not text.strip():  # Check if the text is empty after stripping whitespaces
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=figsize)
        message = "No text available"
        if sentiment_filter:
            message += f" for {sentiment_filter} sentiment"
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
        ax.axis("off")  # Hide the axes
        return fig

    # Generate the word cloud
    # The WordCloud class handles tokenization, word counting, and layout
    wordcloud = WordCloud(
        width=VISUALIZATION["wordcloud"]["width"],  # Width in pixels
        height=VISUALIZATION["wordcloud"]["height"],  # Height in pixels
        max_words=VISUALIZATION["wordcloud"]["max_words"],  # Limit total words shown
        stopwords=all_stopwords,  # Words to exclude
        background_color=VISUALIZATION["wordcloud"][
            "background_color"
        ],  # Background color
        colormap=VISUALIZATION["wordcloud"][
            "colormap"
        ],  # Color scheme for words (viridis is a blue-green-yellow palette)
        contour_width=1,  # Add a border
        contour_color="steelblue",  # Border color
    ).generate(
        text
    )  # This tokenizes text and generates the cloud

    # Create a Matplotlib figure to display the word cloud
    fig, ax = plt.subplots(figsize=figsize)

    # Display the word cloud image
    # imshow() displays data as an image
    # bilinear interpolation makes the image look smoother
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")  # Hide the axes

    # Add a title
    title = "Word Cloud"
    if sentiment_filter:
        title += f" - {sentiment_filter.capitalize()} Sentiment"
    plt.title(title, fontsize=16)

    return fig


def fig_to_base64(fig: Figure) -> str:
    """
    Convert a matplotlib figure to base64 string for embedding in HTML/GUI.

    This is useful for web applications (like Streamlit) where you need to
    embed an image directly in HTML rather than saving to a file.

    Args:
        fig (Figure): Matplotlib figure to convert

    Returns:
        str: Base64 encoded string of the figure (can be used in HTML img tags)
    """
    # Create an in-memory binary buffer
    img_buf = BytesIO()

    # Save the figure to the buffer in PNG format
    # bbox_inches='tight' ensures that the entire figure with labels is included
    fig.savefig(img_buf, format="png", bbox_inches="tight")

    # Move buffer position to the beginning
    img_buf.seek(0)

    # Encode the image data as base64
    # Read the binary data, encode to base64, convert to string
    img_data = base64.b64encode(img_buf.read()).decode("utf-8")

    # Close the figure to free memory
    plt.close(fig)

    return img_data


def create_top_terms_chart(
    data: List[Dict[str, Any]],
    top_n: int = 15,
    sentiment_filter: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 0),
) -> Figure:
    """
    Create a horizontal bar chart of the most frequent terms in the data.

    Args:
        data (List[Dict]): List of dictionaries with Reddit data
        top_n (int): Number of top terms to show
        sentiment_filter (str, optional): Filter by sentiment ('positive', 'negative', 'neutral')
        figsize (Tuple[int, int]): Figure size (width, height) in inches

    Returns:
        Figure: Matplotlib figure object containing the chart
    """
    # Prepare stopwords (words to exclude from the word cloud)
    # Combine standard stopwords with Reddit-specific ones defined earlier
    all_stopwords = set(STOPWORDS).union(REDDIT_STOPWORDS)

    # Collect relevant text based on sentiment filter
    all_text = []
    logging.info(f"Processing {len(data)} items for top terms chart")

    # Process each Reddit item
    for item in data:
        # Skip items that do not match the sentiment filter (if specified)
        if sentiment_filter and "sentiment" in item and "combined" in item["sentiment"]:
            if item["sentiment"]["combined"]["sentiment"] != sentiment_filter:
                # Only allow items which label matches the filter specified
                continue

        # Handle posts - 'title' and 'body'
        if "title" in item:
            # Add the title text
            all_text.append(item["title"])

            # Add the post content if it exists
            if "selftext" in item and item["selftext"]:
                all_text.append(item["selftext"])

        # Handle comments - 'body'
        elif "body" in item and item["body"]:
            all_text.append(item["body"])

    # Debug info
    logging.info(f"Collected {len(all_text)} text segments for analysis")

    # Combine all text into a single string
    # join() concatenates all strings in all_text with a space between them
    text = " ".join(all_text)

    # Handle the case where no text is available
    if not text.strip():  # Check if the text is empty after stripping whitespaces
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=figsize)
        message = "No text available"
        if sentiment_filter:
            message += f" for {sentiment_filter} sentiment"
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
        ax.axis("off")  # Hide the axes
        return fig

    # Tokenize text into words
    # findall() finds all occurences of a pattern in a string
    # The pattern \b\w+\b matches word boundaries with word characters in between
    # i.e. it matches whole words, ignores punctuation
    words = re.findall(r"\b\w+\b", text.lower())
    logging.info(f"Found {len(words)} words in total")

    # Remove stopwords and very short words
    filtered_words = [
        word for word in words if word not in all_stopwords and len(word) > 2
    ]
    logging.info(f"After filtering: {len(filtered_words)} words remaining")

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Get top N terms
    # most_common(n) returns the n most common elements and their counts
    top_terms = word_counts.most_common(top_n)
    logging.info(f"Top {len(top_terms)} terms: {top_terms[:5]}...")

    # Create DataFrame for plotting
    # Convert the list of tuples from Counter to a pandas DataFrame
    df = pd.DataFrame(top_terms, columns=["Term", "Frequency"])

    # Check if we have any terms
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        message = "No significant terms found"
        if sentiment_filter:
            message += f" for {sentiment_filter} sentiment"
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate minimum height needed (prevent layout issues)
    min_height = (
        len(df) * 0.4 + 2
    )  # Each term needs about 0.4 inches of height plus margins

    # If needed height exceeds figure height, adjust figure size
    if min_height > figsize[1]:
        plt.close(fig)  # Close the initial figure
        logging.info(f"Adjusting figure height from {figsize[1]} to {min_height}")
        fig, ax = plt.subplots(figsize=(figsize[0], min_height))

    # Plot horizontal bars (in descending order)
    # [::-1]  reverses the order to show most frequent terms at the top
    # barh() creates a horizontal bar chart
    bars = ax.barh(df["Term"][::-1], df["Frequency"][::-1], color="#2196F3")

    # Add frequency labels at the end of each bar
    for bar in bars:
        # For each bar, get its width (i.e. the frequency value)
        width = bar.get_width()
        # Add text to the right of the bar end
        ax.text(
            width + 0.5,  # x position (just after bar end)
            bar.get_y() + bar.get_height() / 2,  # y position (middle of bar)
            f"{width}",  # text (the frequency)
            ha="left",  # horizontal alignment left
            va="center",  # vertical alignment center
            fontsize=10,
        )  # font size)

    # Set title based on sentiment filter
    title = f"Top {top_n} Terms"
    if sentiment_filter:
        title += f" - {sentiment_filter.capitalize()} Sentiment"
    ax.set_title(title, fontsize=16)

    # Set label for axes
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Term", fontsize=12)

    # Remove Top and right spines for cleaner look
    # Spines are the lines that connect the axis tick marks
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_subreddit_comparison(
    data: Dict[str, List[Dict[str, Any]]], figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Create a comparison chart of sentiment across different subreddits.

    Args:
        data (Dict[str, List[Dict]]): Dictionary mapping subreddit names to lists of analyzed data
        figsize (Tuple[int, int]): Figure size (width, height) in inches

    Returns:
        Figure: Matplotlib figure object containing the comparison chart
    """
    # Prepare data for plotting
    subreddits = []  # List to store subreddit names
    pos_percentages = []  # List to store positive percentages
    neu_percentages = []  # List to store neutral percentages
    neg_percentages = []  # List to store negative percentages
    avg_scores = []  # List to store average sentiment scores

    # Process each subreddit's data
    for subreddit, items in data.items():
        # Skip empty data
        if not items:
            continue

        # Get sentiment labels for this subreddit
        sentiments = [
            item["sentiment"]["combined"]["sentiment"]
            for item in items
            if "sentiment" in item and "combined" in item["sentiment"]
        ]

        # Get sentiment scores for this subreddit
        scores = [
            item["sentiment"]["combined"]["score"]
            for item in items
            if "sentiment" in item and "combined" in item["sentiment"]
        ]

        # Skip if no sentiment data found
        if not sentiments:
            continue

        # Count sentiment occurences
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)

        # Calculate percentages for each sentiment type
        pos_pct = sentiment_counts.get("positive", 0) / total * 100  # % positive
        neu_pct = sentiment_counts.get("neutral", 0) / total * 100  # % neutral
        neg_pct = sentiment_counts.get("negative", 0) / total * 100  # % negative

        # Calculate average sentiment score
        avg_score = sum(scores) / len(scores) if scores else 0

        # Append data to lists for plotting
        subreddits.append(subreddit)
        pos_percentages.append(pos_pct)
        neu_percentages.append(neu_pct)
        neg_percentages.append(neg_pct)
        avg_scores.append(avg_score)

    # Handle case with no data
    if not subreddits:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No subreddit data available for comparison",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Create figure with two subplots
    # First subplot for sentiment percentages, second for average scores
    fig, (ax1, ax2) = plt.subplots(
        2,  # 2 rows
        1,  # 1 column
        figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1]},  # First plot twice as tall
    )

    # Plot stacked percentages bar chart in first subplot
    bar_width = 0.6  # Width of the bars
    x = np.arange(len(subreddits))  # X positions for bars

    # First, plot the positive percentages at the bottom
    ax1.bar(x, pos_percentages, bar_width, label="Positive", color="#4CAF50")

    # Then plot neutral percentages on top of positive percentages
    ax1.bar(
        x,
        neu_percentages,
        bar_width,
        bottom=pos_percentages,
        label="Neutral",
        color="#9E9E9E",
    )

    # Calculate the bottom positions for negative percentages
    # (needs to be on top of both positive and neutral)
    bottom_values = [p + n for p, n in zip(pos_percentages, neu_percentages)]

    # Plot negative percentages on top
    ax1.bar(
        x,
        neg_percentages,
        bar_width,
        bottom=bottom_values,
        label="Negative",
        color="#F44336",
    )

    # Add labels and title to first subplot
    ax1.set_title("Sentiment Distribution by Subreddit", fontsize=16)
    ax1.set_ylabel("Percentage", fontsize=12)
    ax1.set_xticks(x)  # Set x-tick positions
    ax1.set_xticklabels(subreddits, rotation=45, ha="right")  # Set x-tick labels
    ax1.legend(loc="upper right")  # Add legend

    # Add percentage label on bars
    for i, (pos, neu, neg) in enumerate(
        zip(pos_percentages, neu_percentages, neg_percentages)
    ):
        # Only add labels if the percentage is significant (> 5%)
        if pos > 5:
            # Add label in the middle of the positive section
            ax1.text(
                i,
                pos / 2,
                f"{pos:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
        if neu > 5:
            # Add label in the middle of the neutral section
            ax1.text(
                i,
                pos + neu / 2,
                f"{neu:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )
        if neg > 5:
            # Add label in the middle of the negative section
            ax1.text(
                i,
                pos + neu + neg / 2,
                f"{neg:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    # Plot average sentiment score in second subplot
    # Create bar chart of average sentiment scores for each subreddit
    ax2.bar(x, avg_scores, bar_width, color="#2196F3")

    # Add reference line and color regions
    # Add horizontal line at y=0 (neutral sentiment)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    # Add colored background regions to indicate sentiment ranges
    # Light green background for positive sentiment range
    ax2.axhspan(
        SENTIMENT_ANALYSIS["positive_threshold"], 1, facecolor="#E8F5E9", alpha=0.3
    )
    # Light gray background for neutral sentiment range
    ax2.axhspan(
        SENTIMENT_ANALYSIS["negative_threshold"],
        SENTIMENT_ANALYSIS["positive_threshold"],
        facecolor="#EEEEEE",
        alpha=0.3,
    )
    # Light red background for negative sentiment range
    ax2.axhspan(
        -1, SENTIMENT_ANALYSIS["negative_threshold"], facecolor="#FFEBEE", alpha=0.3
    )

    # Set y-axis limits to the sentiment score range
    ax2.set_ylim(-1, 1)  # Sentiment scores range from -1 to 1

    # Add labels to second subplot
    ax2.set_ylabel("Avg. Score", fontsize=12)
    ax2.set_xticks(x)  # Set x-tick positions
    ax2.set_xticklabels(subreddits, rotation=45, ha="right")  # Set x-tick labels

    # Add score labels on bars
    for i, score in enumerate(avg_scores):
        # Determine text color based on background (for readability)
        text_color = "black" if -0.3 < score < 0.3 else "white"

        # Add text label with the score value
        # Position slightly above bar if positive, below if negative
        ax2.text(
            i,
            score + (0.1 if score > 0 else -0.1),
            f"{score:.2f}",  # Format to 2 decimal places
            ha="center",  # Horizontal alignment center
            va="center",  # Vertical alignment center
            color=text_color,
            fontweight="bold",
        )

    # Adjust layout so everything fits
    plt.tight_layout()

    return fig


# Example usage
if __name__ == "__main__":
    # Create more diverse sample data with better time distribution
    from datetime import datetime, timedelta

    # Configure more verbose logging to debug issues
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Base time for our sample data (5 days ago)
    base_time = datetime.now() - timedelta(days=5)

    # Create more diverse sample data with balanced sentiments
    sample_data = [
        {
            "id": "post1",
            "title": "I absolutely love Python programming!",
            "selftext": "The community is amazing and supportive.",
            "created_utc": (base_time + timedelta(days=0)).timestamp(),  # 5 days ago
            "sentiment": {"combined": {"sentiment": "positive", "score": 0.75}},
        },
        {
            "id": "post2",
            "title": "Having issues with my code",
            "selftext": "Nothing works and I'm getting frustrated.",
            "created_utc": (base_time + timedelta(days=1)).timestamp(),  # 4 days ago
            "sentiment": {"combined": {"sentiment": "negative", "score": -0.65}},
        },
        {
            "id": "post3",
            "title": "Just started learning Python",
            "selftext": "I'm neither excited nor disappointed yet, just learning the basics.",
            "created_utc": (base_time + timedelta(days=2)).timestamp(),  # 3 days ago
            "sentiment": {"combined": {"sentiment": "neutral", "score": 0.05}},
        },
        {
            "id": "post4",
            "title": "This framework is terrible and buggy",
            "selftext": "I'm really disappointed with the quality and documentation.",
            "created_utc": (
                base_time + timedelta(days=2, hours=12)
            ).timestamp(),  # 3.5 days ago
            "sentiment": {"combined": {"sentiment": "negative", "score": -0.80}},
        },
        {
            "id": "post5",
            "title": "Python vs JavaScript for beginners",
            "selftext": "I think Python has a gentler learning curve, but JavaScript is more versatile.",
            "created_utc": (base_time + timedelta(days=3)).timestamp(),  # 2 days ago
            "sentiment": {"combined": {"sentiment": "neutral", "score": 0.1}},
        },
        {
            "id": "post6",
            "title": "I'm really struggling with these concepts",
            "selftext": "The documentation is confusing and examples don't work.",
            "created_utc": (
                base_time + timedelta(days=3, hours=12)
            ).timestamp(),  # 2.5 days ago
            "sentiment": {"combined": {"sentiment": "negative", "score": -0.55}},
        },
        {
            "id": "post7",
            "title": "Just built my first web scraper!",
            "selftext": "So proud of myself for building this tool that actually works!",
            "created_utc": (base_time + timedelta(days=4)).timestamp(),  # 1 day ago
            "sentiment": {"combined": {"sentiment": "positive", "score": 0.8}},
        },
    ]

    # Print info about the test data
    sentiments = [item["sentiment"]["combined"]["sentiment"] for item in sample_data]
    sentiment_counts = {
        sentiment: sentiments.count(sentiment) for sentiment in set(sentiments)
    }
    print(f"Test data sentiment distribution: {sentiment_counts}")

    # Test distribution chart
    print("\nTesting plot_sentiment_distribution...")
    fig1 = plot_sentiment_distribution(sample_data)
    plt.figure(fig1.number)
    plt.show()

    # Test time series chart with sentiment trends
    print("\nTesting plot_sentiment_over_time...")
    fig2 = plot_sentiment_over_time(sample_data)
    plt.figure(fig2.number)
    plt.show()

    # Test word cloud
    print("\nTesting generate_wordcloud...")
    fig3 = generate_wordcloud(sample_data)
    plt.figure(fig3.number)
    plt.show()

    # Test word cloud with sentiment filtering
    print("\nTesting generate_wordcloud with positive sentiment filter...")
    fig4 = generate_wordcloud(sample_data, sentiment_filter="positive")
    plt.figure(fig4.number)
    plt.show()

    # Test top terms chart with adjusted parameters
    print("\nTesting create_top_terms_chart...")
    try:
        # Add debug info
        print(f"Sample data length: {len(sample_data)}")
        text_count = sum(1 for item in sample_data if "title" in item or "body" in item)
        print(f"Items with text content: {text_count}")

        # Increase figure size for top terms chart
        fig5 = create_top_terms_chart(sample_data, top_n=10, figsize=(12, 10))
        plt.figure(fig5.number)
        plt.show()
    except Exception as e:
        print(f"Error in create_top_terms_chart: {e}")
        import traceback

        traceback.print_exc()

    # Test subreddit comparison
    print("\nTesting plot_subreddit_comparison...")
    try:
        subreddit_data = {
            "python": sample_data[:3],
            "technology": sample_data[3:5],
            "programming": sample_data,
        }
        fig6 = plot_subreddit_comparison(subreddit_data)
        plt.figure(fig6.number)
        plt.show()
    except Exception as e:
        print(f"Error in plot_subreddit_comparison: {e}")
        import traceback

        traceback.print_exc()

    print("\nAll visualization tests completed.")

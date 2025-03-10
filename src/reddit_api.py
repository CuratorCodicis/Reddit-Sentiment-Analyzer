import os  # Used to access environment variables (Reddit API keys)
import time
import logging
import praw  # The Python Reddit API Wrapper (for fetching data)
from typing import List, Dict, Any, Tuple, Optional, Union  # For type hints

from database import insert_documents
from config import REDDIT_API, RATE_LIMIT


# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id=REDDIT_API["client_id"],
    client_secret=REDDIT_API["client_secret"],
    user_agent=REDDIT_API["user_agent"],
)

# Simple rate-limiting constant (in seconds)
REQUEST_DELAY = RATE_LIMIT["request_delay"]


def fetch_posts(
    subreddit_name: str, limit: int = 10, after: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetches the top 'limit' posts from a given subreddit.

    Args:
        subreddit_name: Name of the subreddit (without the 'r/')
        limit: Number of posts to fetch
        after: Reddit ID to fetch posts after (for pagination)

    Returns:
        List of post dictionaries
    """

    posts = []
    try:
        # Access the subreddit
        subreddit = reddit.subreddit(subreddit_name)

        try:
            _ = subreddit.description  # This forces PRAW to verify if the subreddit exists
        except Exception as e:
            logging.warning(f"❌ Subreddit r/{subreddit_name} does not exist! (Error: {e})")
            return None

        # Get 'hot' posts up to 'limit'
        # If 'after' is provided, it's used for pagination
        params = {"after": f"t3_{after}"} if after else None

        for post in subreddit.hot(limit=limit, params=params):
            # Do not include stickied posts as they skew  the historical data
            if getattr(post, "stickied", False):
                continue
            posts.append(
                {
                    "id": post.id,
                    "title": post.title,
                    "url": post.url,
                    "author": post.author.name if post.author else "Unknown",
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "selftext": post.selftext,
                    "subreddit": subreddit_name.lower(),
                }
            )

    except Exception as e:
        logging.error(f"Error fetching posts from r/{subreddit_name}: {e}")

    logging.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")

    return posts  # Return the list of post dictionaries


def fetch_comments(post_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetches the top 'limit' comments from a given Reddit post.

    Args:
        post_id: The ID of the Reddit post
        limit: Maximum number of comments to fetch

    Returns:
        List of dictionaries containing comment details
    """
    comments = []
    try:
        submission = reddit.submission(id=post_id)  # Fetch the post using its ID
        submission.comments.replace_more(
            limit=0
        )  # Remove "load more comments" placeholders

        # Get top comments, sorted by score
        # Convert to list and sort by score to get the most relevant comments
        all_comments = list(submission.comments.list())
        all_comments.sort(key=lambda c: c.score, reverse=True)

        # Take top N comments
        for comment in all_comments[:min(limit, len(all_comments))]:
            # Skip comments that are deleted or removed
            if comment.author is None or comment.body in ['[deleted]', '[removed]']:
                continue
            
            comments.append(
                {
                    "id": comment.id,
                    "author": comment.author.name if comment.author else "Unknown",
                    "body": comment.body,
                    "score": comment.score,
                    "created_utc": comment.created_utc,
                    "post_id": post_id,
                    "subreddit": submission.subreddit.display_name.lower(),
                }
            )
    except Exception as e:
        logging.error(f"Error fetching comments for post {post_id}: {e}")

    logging.info(f"Fetched {len(comments)} comments for post {post_id}")

    return comments  # Return the list of comment dictionaries


def fetch_and_store_subreddit(subreddit_name, post_limit=5, comment_limit=5):
    """
    Fetch 'hot' posts and thier top comments from a subreddit and store them in MongoDB.
    This function includes a small rate-limit delay to avoid spamming the Reddit API.

    :param subreddit_name: The name of the subreddit (e.g., 'technology').
    :param post_limit: Number of 'hot' posts to fetch (default: 5).
    :param comment_limit: Number of 'top' comments to fetch for each post (default: 5)
    """
    logging.info(f"Preparing to fetch {post_limit} posts from r/{subreddit_name}...")

    # Rate limiting: wait a bit before making the API call
    time.sleep(REQUEST_DELAY)

    # Fetch posts from subreddit
    posts = fetch_posts(subreddit_name, post_limit)
    if not posts:
        logging.warning(f"No posts returned from r/{subreddit_name}.")
        return

    # Insert posts into "post" collection in MongoDB
    inserted_ids = insert_documents("posts", posts)
    if inserted_ids:
        logging.info(
            f"Inserted {len(inserted_ids)} posts from r/{subreddit_name} into MongoDB."
        )
    else:
        logging.warning(f"Failed to insert documents for r/{subreddit_name}.")

    # Fetch and store comments for each post
    all_comments = []
    for post in posts:
        post_id = post["id"]
        # Rate limit for fetching comments
        time.sleep(REQUEST_DELAY)
        comments = fetch_comments(post_id, comment_limit)
        logging.info(f"Fetched {len(comments)} comments for post ID {post_id}.")

        # Associate each comment with its post
        for comment in comments:
            comment["post_id"] = post_id
        all_comments.extend(comments)

    if all_comments:
        inserted_comment_ids = insert_documents("comments", all_comments)
        if inserted_comment_ids:
            logging.info(
                f"Inserted {len(inserted_comment_ids)} comments for posts from r/{subreddit_name} into MongoDB."
            )
        else:
            logging.warning(f"Failed to insert comments for r/{subreddit_name}.")

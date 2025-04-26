"""
Streamlit Web Interface for Reddit Sentiment Analyzer

This module creates a Streamlit web application that follows this workflow:
1. Check if data exists in MongoDB
2. If insufficient data, fetch from Reddit API
3. Preprocess and store in MongoDB
4. Analyze sentiment of the retrieved data
5. Create and display visualizations

Run with: streamlit run app.py
"""

import time  # For adding delays between API requests
import logging  # For logging information and errors
from datetime import datetime, timedelta  # For timestamp manipulation and formatting
from typing import List, Dict, Any, Tuple, Optional, Callable, Union  # For type hints
import random

import pandas as pd  # For data manipulation and analysis
import streamlit as st  # For creating the web interface
import matplotlib.pyplot as plt  # For plotting (used by visualization module)

from database import (
    fetch_documents,  # For retrieving data from MongoDB
    insert_documents,  # For storing data in MongoDB
    get_mongo_client,  # For checking MongoDB connection
)
from reddit_api import (
    fetch_posts,  # For fetching posts from Reddit API
    fetch_comments,  # For fetching comments from Reddit API
)
from data_preprocessing import (
    preprocess_data,  # For cleaning and preprocessing text data
    filter_by_keywords,  # For filtering posts/comments based on keywords
)
from sentiment_analysis import SentimentAnalyzer  # For sentiment analysis
from visualization import (
    plot_sentiment_distribution,  # For creating sentiment pie charts
    plot_sentiment_over_time,  # For creating time-series sentiment charts
    generate_wordcloud,  # For generating word clouds
    create_top_terms_chart,  # For creating term frequency charts
    plot_subreddit_comparison, # For comparing the sentiment of different subreddits
)
from llm_utils import process_items, topic_analysis
from config import (
    SAMPLE_SUBREDDITS,
    VISUALIZATION,
    SENTIMENT_ANALYSIS,
    DEFAULT_SETTINGS,
    RATE_LIMIT,
    DOCKERIZED,
    LLM
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class RedditSentimentApp:
    """
    Main application class for the Reddit Sentiment Analyzer.
    
    This class handles all aspects of the Streamlit application including:
    - UI building and rendering
    - Data fetching and processing
    - Sentiment analysis
    - Visualization creation
    - State management
    """
    
    def __init__(self):
        """
        Initialize the application, set up configuration and state.
        
        This initializes the page configuration, session state variables,
        and loads the sentiment analyzer.
        """
        self.setup_page_config()
        self.initialize_session_state()
        self.sentiment_analyzer = self.get_sentiment_analyzer()
    
    def setup_page_config(self):
        """
        Configure Streamlit page settings.
        
        Sets the page title, icon, layout, and sidebar state.
        """
        st.set_page_config(
            page_title="Reddit Sentiment Analyzer",
            page_icon="📊",
            layout="centered",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """
        Initialize all session state variables if they don't exist.
        
        This ensures all required state variables are available when the app runs.
        """
        # Results storage
        if "analyzed_posts" not in st.session_state:
            st.session_state.analyzed_posts = []
        if "analyzed_comments" not in st.session_state:
            st.session_state.analyzed_comments = []
        if "posts" not in st.session_state:
            st.session_state.posts = []
        if "comments" not in st.session_state:
            st.session_state.comments = []
            
        # Input settings
        if "subreddit" not in st.session_state:
            st.session_state.subreddit = None
        if "post_limit" not in st.session_state:
            st.session_state.post_limit = DEFAULT_SETTINGS["post_limit"]
        if "include_comments" not in st.session_state:
            st.session_state.include_comments = False
        if "comment_limit" not in st.session_state:
            st.session_state.comment_limit = DEFAULT_SETTINGS["comment_limit"]
        if "filter_keywords" not in st.session_state:
            st.session_state.filter_keywords = False
        if "keywords" not in st.session_state:
            st.session_state.keywords = None
            
        # UI state
        if "sentiment_filter" not in st.session_state:
            st.session_state.sentiment_filter = "All"
        if "term_count" not in st.session_state:
            st.session_state.term_count = 10
        if 'has_analyzed' not in st.session_state:
            st.session_state.has_analyzed = False
            
        # Status tracking
        if 'fetch_status' not in st.session_state:
            st.session_state.fetch_status = ""
        if 'analysis_status' not in st.session_state:
            st.session_state.analysis_status = ""
        
        # Subreddit comparison state
        if "analyzed_subreddits" not in st.session_state:
            st.session_state.analyzed_subreddits = {}
        if "comparison_enabled" not in st.session_state:
            st.session_state.comparison_enabled = False
        if "comparison_subreddit" not in st.session_state:
            st.session_state.comparison_subreddit = None
    
    @st.cache_resource
    def get_sentiment_analyzer(_self) -> SentimentAnalyzer:
        """
        Create and cache a SentimentAnalyzer instance.
        
        Returns:
            SentimentAnalyzer: An instance of our sentiment analyzer class
        """
        return SentimentAnalyzer()
    
    def build_header(self):
        """
        Create the application header with logo, title, and technology labels.
        
        Returns:
            The container holding the header elements
        """
        header_container = st.container()
        with header_container:
            cols = st.columns([1, 6])
            with cols[0]:
                st.markdown(
                    """
                    <style>
                        .reddit-logo {
                            padding-top: 0px; /* Adjust this value to move the logo down */
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.image(
                    "https://www.redditstatic.com/desktop2x/img/favicon/android-icon-192x192.png",
                    output_format="PNG",
                    width=80,
                )
            with cols[1]:
                st.markdown(
                    """
                <h1>Reddit Sentiment Analyzer</h1>
                <p style="font-size: 1.2rem; margin-top: -15px; color: #888;">
                Analyze sentiment patterns in Reddit discussions
                </p>
                """,
                    unsafe_allow_html=True,
                )

            # Define technology labels
            tech_labels = [
                ("Python", "#306998"),
                ("PRAW", "#FF4500"),
                ("NLTK", "#4CAF50"),
                ("TextBlob", "#9933CC"),
                ("VADER", "#4DB6AC"),
                ("Matplotlib", "#11557c"),
                ("Pandas", "#76b900"),
                ("MongoDB", "#FF5722"),
                ("Streamlit", "#FF9800"),
            ]

            # Add Docker label if running in a Docker container
            if DOCKERIZED:
                tech_labels.append(("Docker", "#0db7ed"))

            # Generate HTML for labels
            label_html = " ".join(
                [
                    f"""<span style="background-color: {color}; 
                                    color: white; 
                                    padding: 3px 10px; 
                                    border-radius: 12px; 
                                    font-size: 0.8rem;">
                        {name}
                    </span>"""
                    for name, color in tech_labels
                ]
            )

            # Render the labels properly
            st.markdown(
                f"""
                <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: -10px; padding: 20px 0;">
                    {label_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        return header_container
    
    def build_sidebar(self):
        """
        Build the sidebar UI and collect user inputs.
        
        Returns:
            dict: A dictionary containing all the user input values
        """
        st.sidebar.header("⚙️ Analysis Settings")

        # Subreddit selection
        selected_option = st.sidebar.selectbox(
            "Select subreddit",
            ["Enter custom subreddit..."] + SAMPLE_SUBREDDITS,
            index=(
                0
                if st.session_state.subreddit not in SAMPLE_SUBREDDITS
                else SAMPLE_SUBREDDITS.index(st.session_state.subreddit) + 1
            ),
        )

        if selected_option == "Enter custom subreddit...":
            subreddit = st.sidebar.text_input(
                "Enter subreddit name", value=st.session_state.subreddit or ""
            )
        else:
            subreddit = selected_option

        # Update session state with the selected subreddit
        st.session_state.subreddit = subreddit

        # Additional settings
        keywords_flag = st.sidebar.checkbox(
            "Filter posts for keywords",
            value=st.session_state.filter_keywords,
            help="Filter posts containing these keywords, other posts from the subreddit will be discarded",
        )
        st.session_state.filter_keywords = keywords_flag

        # Keywords for filtering posts
        if keywords_flag:
            keywords = st.sidebar.text_input(
                "Keywords (comma separated)", value=st.session_state.keywords or ""
            )
        else:
            keywords = None
        st.session_state.keywords = keywords

        # For the Include comments checkbox:
        fetch_comments_flag = st.sidebar.checkbox(
            "Include comments",
            value=st.session_state.include_comments,
            help="Also fetch and analyze comments (slower)",
        )
        st.session_state.include_comments = fetch_comments_flag

        st.sidebar.markdown("---")

        # Number of posts to analyze with improved UX
        st.sidebar.subheader("Data Collection")

        # Use container for post limit controls
        post_limit_container = st.sidebar.container()
        post_limit_container.markdown("##### Number of posts to analyze")

        # First row: Preset buttons for quick selection
        preset_cols = post_limit_container.columns(4)
        if preset_cols[0].button("25", use_container_width=True):
            st.session_state.post_limit = 25
        if preset_cols[1].button("50", use_container_width=True):
            st.session_state.post_limit = 50
        if preset_cols[2].button("100", use_container_width=True):
            st.session_state.post_limit = 100
        if preset_cols[3].button("250", use_container_width=True):
            st.session_state.post_limit = 250

        # Second row: Slider and number input
        post_limit_cols = post_limit_container.columns([3, 1])

        with post_limit_cols[0]:
            # Slider with increased maximum value
            post_limit = st.slider(
                "Adjust post count",
                min_value=10,
                max_value=500,
                value=st.session_state.post_limit,
                step=10,
                label_visibility="collapsed",
            )

        with post_limit_cols[1]:
            # Number input for precise control
            post_limit = st.number_input(
                "Exact count",
                min_value=10,
                max_value=1000,  # Allow even more posts for power users
                value=post_limit,
                step=10,
                label_visibility="collapsed",
            )

        st.session_state.post_limit = post_limit

        # Add a warning when large post counts are selected
        if post_limit > 200:
            st.sidebar.warning(f"⚠️ Fetching {post_limit} posts may take longer to process.")

        # Comments per post (only shown if comments are enabled)
        comment_limit = 0
        if fetch_comments_flag:
            comment_limit_container = st.sidebar.container()
            comment_limit_container.markdown("##### Comments per post")
            
            # First row: Preset buttons for comments
            comment_preset_cols = comment_limit_container.columns(4)
            if comment_preset_cols[0].button("5", key="comment_5", use_container_width=True):
                st.session_state.comment_limit = 5
            if comment_preset_cols[1].button("10", key="comment_10", use_container_width=True):
                st.session_state.comment_limit = 10
            if comment_preset_cols[2].button("20", key="comment_20", use_container_width=True):
                st.session_state.comment_limit = 20
            if comment_preset_cols[3].button("30", key="comment_30", use_container_width=True):
                st.session_state.comment_limit = 30
            
            # Second row: slider and number input for comments
            comment_limit_cols = comment_limit_container.columns([3, 1])
            
            with comment_limit_cols[0]:
                comment_limit = st.slider(
                    "Adjust comment count",
                    min_value=5,
                    max_value=100,  # Increased max comments
                    value=st.session_state.comment_limit,
                    step=5,
                    label_visibility="collapsed",
                )
            
            with comment_limit_cols[1]:
                comment_limit = st.number_input(
                    "Exact comment count",
                    min_value=5,
                    max_value=200,  # Allow more comments for power users
                    value=comment_limit,
                    step=5,
                    label_visibility="collapsed",
                )
            
            st.session_state.comment_limit = comment_limit
            
            # Add a warning when large comment counts are selected
            if comment_limit > 50:
                st.sidebar.info("ℹ️ High comment counts may significantly increase processing time.")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Subreddit Comparison")
        
        # Enable comparison mode
        comparison_enabled = st.sidebar.checkbox(
            "Compare with another subreddit", 
            value=st.session_state.comparison_enabled,
            help="Compare sentiment with another subreddit"
        )
        st.session_state.comparison_enabled = comparison_enabled
        
        if comparison_enabled:
            # Let user select comparison subreddit
            comparison_subreddit = st.sidebar.text_input(
                "Enter comparison subreddit",
                value=st.session_state.comparison_subreddit or "",
                placeholder="e.g., programming"
            )
            st.session_state.comparison_subreddit = comparison_subreddit

        st.sidebar.markdown("---")
        # Analyze button to trigger processing
        if not subreddit:
            st.sidebar.button("🧐 Analyze Subreddit", on_click=self.on_analyze_click, disabled=True, use_container_width=True)
            st.sidebar.warning("Please enter a subreddit before analyzing.")
        else:
            st.sidebar.button("🧐 Analyze Subreddit", on_click=self.on_analyze_click, use_container_width=True)

        # Clear analysis button to reset
        if st.session_state.has_analyzed:
            if st.sidebar.button("🧹 Clear Analysis", use_container_width=True):
                self.clear_analysis()

        # Return collected inputs for convenience
        return {
            'subreddit': subreddit,
            'keywords': keywords,
            'post_limit': post_limit,
            'comment_limit': comment_limit,
            'fetch_comments_flag': fetch_comments_flag
        }
    
    def on_analyze_click(self):
        """
        Handle the analyze button click event.
        
        Fetches, processes, and analyzes data, then updates session state.
        """
        st.session_state.has_analyzed = True
        
        subreddit = st.session_state.subreddit
        keywords = st.session_state.keywords
        post_limit = st.session_state.post_limit
        comment_limit = st.session_state.comment_limit
        fetch_comments_flag = st.session_state.include_comments
        comparison_enabled = st.session_state.comparison_enabled
        comparison_subreddit = st.session_state.comparison_subreddit

        # Create progress bars
        progress_bars, headers = self.create_progress_bars(
            show_comments=fetch_comments_flag,
            show_comparison=comparison_enabled and comparison_subreddit
        )
        
        # Define progress callback
        def update_progress(process_name, value, status_text=""):
            if process_name in progress_bars:
                progress_bars[process_name].progress(value, text=status_text)
        
        with st.spinner(f"  🐢", show_time=True):
            analyzed_posts, analyzed_comments, fetch_status = self.fetch_and_process_data(
                subreddit, keywords, post_limit, comment_limit, fetch_comments_flag,
                progress_callback=update_progress
            )

            analysis_status = f"Posts and comments already analyzed during fetch."
            
            # Store all results in session state
            st.session_state.posts = analyzed_posts
            st.session_state.comments = analyzed_comments
            st.session_state.fetch_status = fetch_status
            st.session_state.analyzed_posts = analyzed_posts
            st.session_state.analyzed_comments = analyzed_comments
            st.session_state.analysis_status = analysis_status

            # Also store this subreddit's data for comparison
            if analyzed_posts:  # Only store if we successfully got posts
                st.session_state.analyzed_subreddits[subreddit] = {
                    'posts': analyzed_posts,
                    'comments': analyzed_comments,
                    'fetch_status': fetch_status,
                    'analysis_status': analysis_status
                }
        
            if comparison_subreddit:
                # Create progress callback for comparison
                def update_comparison_progress(process_name, value, status_text=""):
                    comp_process = f"comparison_{process_name}"
                    if comp_process in progress_bars:
                        progress_bars[comp_process].progress(value, text=status_text)

                analyzed_posts, analyzed_comments, fetch_status = self.fetch_and_process_data(
                    comparison_subreddit, keywords, post_limit, comment_limit, fetch_comments_flag,
                    progress_callback=update_comparison_progress
                )
                
                analysis_status = f"Posts and comments already analyzed during fetch."

                # Also store this subreddit's data for comparison
                if analyzed_posts:  # Only store if we successfully got posts
                    st.session_state.analyzed_subreddits[comparison_subreddit] = {
                        'posts': analyzed_posts,
                        'comments': analyzed_comments,
                        'fetch_status': fetch_status,
                        'analysis_status': analysis_status
                    }
        # Short delay to ensure user sees completed progress
        time.sleep(0.5)
        
        # Show completion message
        self.show_completion_message()

        for e in headers:
            e.empty()
        for key in progress_bars:
            progress_bars[key].empty()

    def clear_analysis(self):
        """
        Clear all analysis data from session state and reload the page.
        """
        st.session_state.has_analyzed = False
        st.session_state.posts = []
        st.session_state.comments = []
        st.session_state.analyzed_posts = []
        st.session_state.analyzed_comments = []
        st.session_state.fetch_status = ""
        st.session_state.analysis_status = ""
        st.rerun()
    
    def create_progress_bars(self, show_comments=False, show_comparison=False):
        """
        Create progress bars for the various fetch operations.
        
        Args:
            show_comments: Whether to show progress bars for comments
            show_comparison: Whether to show progress bars for comparison
        
        Returns:
            Dictionary of progress bars
        """
        progress_bars = {}
        headers = []

        subreddit = st.session_state.subreddit
        
        # Select a random starting message
        start_messages = [
            f"🚀 Scanning r/{subreddit} for hot gossip...",
            f"🔍 Searching for insightful discussions in r/{subreddit}...",
            f"📊 Crunching numbers and analyzing emotions of r/{subreddit}...",
            f"💬 Listening to what r/{subreddit} has to say...",
            f"🎭 Scanning r/{subreddit} for emotional drama...",
            f"🛸 Scanning the digital cosmos of r/{subreddit} for intelligent conversations... results may vary.",
            f"📡 Intercepting signals from r/{subreddit}...",
            f"🤖 I'm sorry, Dave, I'm afraid I can't do that... until I finish analyzing r/{subreddit}.",
            f"🧠 Thinking really hard about r/{subreddit}'s sentiment...",
            f"🏜️ The spice must flow... and so must the data from r/{subreddit}...",
            f"📈 Extracting sentiment trends from r/{subreddit}. Numbers don’t lie, but Redditors might.",
            f"⚔️ One does not simply analyze Reddit without encountering drama..."
        ]

        top_header = st.info(random.choice(start_messages))
        headers.append(top_header)
        
        # Main subreddit posts progress bar
        #headers.append(st.markdown("##### Fetching posts"))
        progress_bars['posts'] = st.progress(0, "Getting started...")
        
        # Comments progress bar (if enabled)
        if show_comments:
            #headers.append(st.markdown("##### Fetching comments"))
            progress_bars['comments'] = st.progress(0, "Waiting for posts...")
        
        # Comparison progress bars (if enabled)
        if show_comparison:
            comp_subreddit = st.session_state.comparison_subreddit

            # Select a random starting message
            comp_messages = [
                f"⚖️ Pitting r/{subreddit} against r/{comp_subreddit}... Who wins the battle of opinions?",
                f"📊 Running comparative analysis: How does r/{subreddit} stack up against r/{comp_subreddit}?",
                f"🔍 Sentiment benchmarking in progress for r/{comp_subreddit}...",
                f"🔍 Measuring the emotional temperature of r/{comp_subreddit}.",
                f"💡 Investigating sentiment trends across r/{subreddit} and r/{comp_subreddit}.",
            ]

            second_header = st.info(random.choice(comp_messages))
            headers.append(second_header)
            #headers.append(st.markdown(f"##### Fetching posts from r/{comp_subreddit}"))
            progress_bars['comparison_posts'] = st.progress(0, "Waiting...")
            
            if show_comments:
                #headers.append(st.markdown(f"##### Fetching comments from r/{comp_subreddit}"))
                progress_bars['comparison_comments'] = st.progress(0, "Waiting...")
        
        return progress_bars, headers

    def show_completion_message(self):
        """Display a random completion message."""
        completion_messages = [
            "🚀 Analysis complete! You just uncovered deep Reddit wisdom!",
            "✨ Boom! Sentiment analyzed!",
            "🔮 The data oracle has spoken! Check out the results below!",
            "🎉 All set! Time to explore some juicy insights!"
            "🔬 Analysis completed. We've turned Reddit chatter into structured insights.",
            "🛠️ Mission accomplished! Sentiment data is ready for interpretation.",
            "💡 Data processed! Now go forth and interpret the chaos.",
            "📢 Sentiment trends detected. Now, let's make sense of them."
        ]
        message = st.success(random.choice(completion_messages))
        # Small delay so user can see the message
        time.sleep(2)
        message.empty()

    
    def fetch_and_process_data(
        self,
        subreddit: str,
        keywords: Optional[str] = None,
        post_limit: int = 25,
        comment_limit: int = 5,
        fetch_comments_flag: bool = False,
        use_cached: bool = True,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Main data processing pipeline that implements the core workflow.

        This function handles the entire data fetching and processing workflow:
        1. Check if sufficient data exists in MongoDB
        2. If not, fetch from Reddit API
        3. Preprocess the data (clean text, etc.)
        4. Store processed data in MongoDB for future use
        5. Return the processed data for analysis

        Args:
            subreddit: Name of the subreddit to analyze
            keywords: Comma-separated keywords to filter by
            post_limit: Maximum number of posts to analyze
            comment_limit: Maximum number of comments to fetch per post
            fetch_comments_flag: Whether to fetch and analyze comments
            use_cached: Whether to use cached data from MongoDB if available

        Returns:
            A tuple containing:
                - List of preprocessed Reddit posts
                - List of preprocessed Reddit comments
                - Processing status message
        """
        status_message = ""
        status_message += f"Analyzing r/{subreddit}...\n"

        # Initial progress update
        if progress_callback:
            progress_callback('posts', 0.1, f"Looking for r/{subreddit} data...")

        # Process keywords if provided
        # Convert comma-separated string to list of keywords
        keyword_list: Optional[List[str]] = None
        if keywords and isinstance(keywords, str):
            # Split by comma, strip whitespace, and remove empty strings
            keyword_list = [
                keyword.strip() for keyword in keywords.split(",") if keyword.strip()
            ]
            # If we have valid keywords, log them and add to the status message
            if keyword_list:
                logging.info(f"Using keywords filter: {keyword_list}")
                status_message += f"Using keywords: {', '.join(keyword_list)}\n"

        # Update progress for cached posts check
        if progress_callback and use_cached:
            progress_callback('posts', 0.2, "Checking for cached data...")
        
        cached_posts: List[Dict[str, Any]] = []
        if use_cached:
            # STEP 1: Check if we have data in MongoDB
            status_message += f"Checking for cached data from r/{subreddit}...\n"

            # Build query for MongoDB - search for posts from this subreddit
            query = {"subreddit": subreddit}

            # Try to retrieve posts from MongoDB using database.py's fetch_documents function
            cached_posts = fetch_documents("posts", query)
            status_message += f"Found {len(cached_posts)} cached posts\n"

            if cached_posts:
                fresh_threshold = timedelta(days=1)  # Timedelta option?
                try:
                    # Assume each post has a numeric "created_utc" timestamp
                    newest_time = max(post.get("created_utc", 0) for post in cached_posts)
                    newest_datetime = datetime.fromtimestamp(newest_time)

                    if datetime.now() - newest_datetime < fresh_threshold:
                        status_message += (
                            f"Cache is fresh (newest post at {newest_datetime}).\n"
                        )
                    else:
                        status_message += (
                            f"Cache is stale (newest post at {newest_datetime}).\n"
                        )
                        cached_posts = []  # Force fetch of fresh data
                except Exception as e:
                    status_message += f"Error determining cache freshness: {e}\n"

        # STEP 2: Determine if we need to fetch new data
        need_to_fetch = not use_cached or not cached_posts or len(cached_posts) < post_limit

        processed_posts = []

        # STEP 3: Fetch new data if needed
        if need_to_fetch:
            status_message += f"Fetching posts from r/{subreddit}...\n"

            # Update progress when fetching posts
            if progress_callback:
                progress_callback('posts', 0.3, f"Fetching posts from r/{subreddit}...")

            # Track fetching statistic
            total_fetched = 0
            fetch_batch_size = post_limit  # Inital batch size
            max_attempts = 3  # Limit fetch attempts

            # Keep track of the last post ID for pagination
            last_post_id = None

            new_posts: List[Dict[str, Any]] = fetch_posts(subreddit, fetch_batch_size)

            # After fetching initial posts - 40%
            if progress_callback:
                progress_callback('posts', 0.4, f"Retrieved {len(new_posts) if new_posts else 0} posts...")


            # Check if the subreddit was found
            if new_posts is None:
                st.error(f"❌ The subreddit r/{subreddit} could not be found. Please check the name and try again.")
                return [], [], f"Subreddit r/{subreddit} not found."

            # If no posts were returned, also show an error
            if not new_posts:
                st.warning(f"⚠️ No posts found in r/{subreddit}. Try using a different subreddit.")
                return [], [], f"No posts found in r/{subreddit}."

            # Preprocess and store new posts
            if new_posts:
                # Save the last post ID
                last_post_id = new_posts[-1]["id"]
                total_fetched += len(new_posts)

                # Clean and preprocess the posts
                status_message += f"Preprocessing {len(new_posts)} posts...\n"
                batch_processed = preprocess_data(new_posts)

                # Add sentiment analysis before storing
                status_message += f"Analyzing sentiment of {len(batch_processed)} posts...\n"
                batch_processed = self.sentiment_analyzer.analyze_reddit_data(batch_processed)

                # LLM processing
                batch_processed = process_items(batch_processed)

                # Store in database
                insert_documents("posts", batch_processed)
                status_message += f"Stored {len(batch_processed)} posts in MongoDB\n"

                processed_posts.extend(batch_processed)

                if progress_callback:
                    progress_callback('posts', 0.5, f"Processed {len(batch_processed)} posts...")

            attempts = 1
            while (
                len(processed_posts) < post_limit
                and attempts < max_attempts
                and last_post_id
            ):
                attempts += 1

                # Calculate how many more posts we need
                still_needed = post_limit - len(processed_posts)

                # Estimate how many to fetch based on success rate
                if total_fetched > 0 and len(processed_posts) > 0:
                    success_rate = len(processed_posts) / total_fetched
                    fetch_more = min(100, int(still_needed / success_rate * 1.2))
                else:
                    fetch_more = still_needed * 2  # Conservative default

                status_message += f"Fetched {len(processed_posts)} valid posts so far. Need {still_needed} more.\n"
                status_message += (
                    f"Fetching {fetch_more} additional posts after ID {last_post_id}...\n"
                )

                # Fetch next batch with pagination
                additional_posts = fetch_posts(subreddit, fetch_more, after=last_post_id)

                if not additional_posts:
                    status_message += "No more posts available from Reddit.\n"
                    break

                # Update tracking variables
                last_post_id = additional_posts[-1]["id"]
                total_fetched += len(additional_posts)

                # Process this batch
                batch_processed = preprocess_data(additional_posts)
                batch_processed = self.sentiment_analyzer.analyze_reddit_data(batch_processed)

                # LLM processing
                batch_processed = process_items(batch_processed)

                insert_documents("posts", batch_processed)
                status_message += f"Stored {len(batch_processed)} posts in MongoDB\n"

                processed_posts.extend(batch_processed)

                # Update progress based on how many posts we've fetched so far
                if progress_callback:
                    # Calculate percentage of posts fetched (range 0.5-0.9)
                    progress_value = 0.5 + (0.4 * (len(processed_posts) / post_limit))
                    progress_callback('posts', progress_value, f"Fetched {len(processed_posts)}/{post_limit} posts...")

                time.sleep(RATE_LIMIT["request_delay"])

            status_message += (
                f"Fetch complete - got {len(processed_posts)} posts after preprocessing\n"
            )
        else:
            # We're using cached data
            status_message += f"Using {len(cached_posts)} cached posts from MongoDB\n"
            processed_posts = cached_posts

        # STEP 4: Apply keyword filtering if needed
        filtered_posts = processed_posts
        if keyword_list:
            # Use data_preprocessing.py's filter_by_keywords function
            pre_filter_count = len(processed_posts)
            filtered_posts = filter_by_keywords(processed_posts, keyword_list)
            status_message += f"Filtered cached data from {pre_filter_count} to {len(filtered_posts)} posts\n"

        # Ensure we limit to requested number of posts
        # Sort by created_utc (newest first) if available
        sorted_posts = filtered_posts
        if filtered_posts and "created_utc" in filtered_posts[0]:
            sorted_posts = sorted(
                filtered_posts, key=lambda x: x.get("created_utc", 0), reverse=True
            )

        # Take only the number of posts requested
        posts: List[Dict[str, Any]] = sorted_posts[:post_limit]

        # After fetch complete - 100%
        if progress_callback:
            progress_callback('posts', 1.0, "Posts completed!")

        # STEP 5: Handle comments if requested
        comments: List[Dict[str, Any]] = []
        if fetch_comments_flag and posts:
            # Start comment progress
            if progress_callback:
                progress_callback('comments', 0.1, "Preparing to fetch comments...")

            # Get all post IDs
            post_ids: List[str] = [post["id"] for post in posts]

            # Check for cached comments first if using cache
            cached_comments = []
            comment_counts_by_post = {}  # Track how many comments each post has in DB
            existing_comment_ids = set()  # Track specific comment IDs we already have

            if use_cached:
                # Query MongoDB for comments from these posts
                comment_query = {"post_id": {"$in": post_ids}}
                cached_comments = fetch_documents("comments", comment_query)
                status_message += f"Found {len(cached_comments)} cached comments\n"
                
                # Count existing comments for each post
                for comment in cached_comments:
                    post_id = comment.get("post_id")
                    comment_id = comment.get("id")

                    if post_id:
                        comment_counts_by_post[post_id] = comment_counts_by_post.get(post_id, 0) + 1
                    if comment_id:
                        existing_comment_ids.add(comment_id)

                if progress_callback:
                    progress_callback('comments', 0.3, f"Found {len(cached_comments)} cached comments...")
        
            # Only fetch comments for posts that have enough comments and are likely to need more
            posts_needing_comments = []
            
            for post in posts:
                post_id = post["id"]
                num_comments = post.get("num_comments", 0)
                cached_count = comment_counts_by_post.get(post_id, 0)
                
                # Skip posts with no comments
                if num_comments == 0:
                    status_message += f"Skipping post {post_id} (0 comments)\n"
                    continue
                    
                # Skip posts where we already have enough cached comments
                if cached_count >= min(comment_limit, num_comments):
                    status_message += f"Skipping post {post_id} (already have {cached_count} of {num_comments} comments)\n"
                    continue
                    
                # Skip posts with very few comments if we have most of them already
                if num_comments <= 3 and cached_count >= num_comments - 1:
                    status_message += f"Skipping post {post_id} (have {cached_count} of only {num_comments} total comments)\n"
                    continue
                
                # This post needs more comments
                posts_needing_comments.append((post, num_comments, cached_count))

            # After determining which posts need comments
            if progress_callback:
                progress_callback('comments', 0.4, f"Fetching comments for {len(posts_needing_comments)} posts...")
    

            # Check if we actually need to fetch any new comments
            if posts_needing_comments:
                status_message += f"Fetching comments for {len(posts_needing_comments)} posts that need more comments...\n"
                
                # Fetch new comments for posts that need more
                new_comments_count = 0
                post_count = 0
                for post, num_comments, cached_count in posts_needing_comments:
                    post_count += 1
                    post_id = post["id"]

                    # Calculate how many more comments we need
                    comments_to_fetch = min(comment_limit - cached_count, num_comments - cached_count)

                    if comments_to_fetch > 0:
                        status_message += f"Fetching {comments_to_fetch} more comments for post {post_id} (has {num_comments} total, {cached_count} cached)\n"
       
                        # Fetch comments for this post
                        post_comments = fetch_comments(post_id, comments_to_fetch + 5)  # Fetch a few extra in case some are already cached
                        
                        if post_comments:
                            # Filter out comments we already have cached
                            new_comments = [c for c in post_comments if c.get("id") not in existing_comment_ids]
                            
                            if new_comments:
                                # Clean and preprocess comments
                                processed_comments = preprocess_data(new_comments)
                                analyzed_comments = self.sentiment_analyzer.analyze_reddit_data(processed_comments)

                                # LLM processing
                                analyzed_comments = process_items(analyzed_comments)

                                # Store enriched comments (with sentiment)
                                insert_documents("comments", analyzed_comments)
                                
                                # Add to our collection
                                comments.extend(analyzed_comments)
                                new_comments_count += len(analyzed_comments)
                                
                                # Add these IDs to our set of existing comment IDs
                                for comment in processed_comments:
                                    existing_comment_ids.add(comment.get("id"))
                            else:
                                status_message += f"All {len(post_comments)} comments for post {post_id} were already cached\n"
                                
                        # Update progress for this post - 40-90%
                        if progress_callback:
                            progress_percent = 0.4 + (0.5 * (post_count / len(posts_needing_comments)))
                            progress_callback('comments', progress_percent, 
                                            f"Fetched comments for {post_count}/{len(posts_needing_comments)} posts...")
                        
                        # Small delay to respect API rate limits
                        time.sleep(RATE_LIMIT["request_delay"])
                
                status_message += f"Fetched {new_comments_count} new comments\n"
            else:
                status_message += "No posts need additional comments\n"

            # Add cached comments to our collection
            comments.extend(cached_comments)
            status_message += f"Using {len(comments)} total comments for analysis\n"

        # After all comments fetched - 100%
        if progress_callback:
            progress_callback('comments', 1.0, f"Comments completed! Found {len(comments)} total")

        # Return the posts and comments
        return posts, comments, status_message

    def analyze_reddit_data(
        self,
        posts: List[Dict[str, Any]],
        comments: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Analyze the sentiment of Reddit posts and comments.

        This function uses the SentimentAnalyzer to determine the sentiment of each post and comment.
        The analyzer adds sentiment information directly to the post/comment dictionaries.

        Args:
            posts: List of preprocessed Reddit posts
            comments: List of preprocessed Reddit comments

        Returns:
            A tuple containing:
                - Posts with sentiment analysis added
                - Comments with sentiment analysis added
                - Analysis status details
        """
        status_message = ""

        # Analyze posts
        status_message += f"Analyzing sentiment of {len(posts)} posts...\n"
        analyzed_posts: List[Dict[str, Any]] = self.sentiment_analyzer.analyze_reddit_data(posts)

        # Analyze comments if available
        analyzed_comments: List[Dict[str, Any]] = []
        if comments:
            status_message += f"Analyzing sentiment of {len(comments)} comments...\n"
            analyzed_comments = self.sentiment_analyzer.analyze_reddit_data(comments)

        return analyzed_posts, analyzed_comments, status_message
    
    def display_sentiment_stats(self, analyzed_posts: List[Dict[str, Any]]):
        """
        Display sentiment statistics and metrics.

        This function calculates and displays metrics about the sentiment
        distribution in the analyzed posts.

        Args:
            analyzed_posts: List of posts with sentiment analysis
        """
        # Extract sentiment information from posts
        sentiments: List[str] = [
            post["sentiment"]["combined"]["sentiment"]
            for post in analyzed_posts
            if "sentiment" in post and "combined" in post["sentiment"]
        ]

        # Count occurrences of each sentiment type
        pos_count = sentiments.count("positive")
        neu_count = sentiments.count("neutral")
        neg_count = sentiments.count("negative")
        total = len(sentiments)

        # Display metrics only if we have data
        if total > 0:
            # Calculate percentages
            pos_pct: float = (pos_count / total) * 100
            neu_pct: float = (neu_count / total) * 100
            neg_pct: float = (neg_count / total) * 100

            # Calculate average sentiment score
            scores: List[float] = [
                post["sentiment"]["combined"]["score"]
                for post in analyzed_posts
                if "sentiment" in post and "combined" in post["sentiment"]
            ]

            if scores:
                avg_score: float = sum(scores) / len(scores)

                # Determine overall sentiment category
                if avg_score >= SENTIMENT_ANALYSIS["positive_threshold"]:
                    sentiment_category: str = "Positive"
                    color: str = "green"
                    emoji: str = "🟢"
                elif avg_score <= SENTIMENT_ANALYSIS["negative_threshold"]:
                    sentiment_category: str = "Negative"
                    color: str = "red"
                    emoji: str = "🔴"
                else:
                    sentiment_category: str = "Neutral"
                    color: str = "gray"
                    emoji: str = "⚪"

                # Display overall sentiment with colored text
                st.markdown(
                    f"""
                    <div style='
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 5px solid {color};
                        margin: 10px 0;'>
                        <h3 style='margin:0;'>Overall Post Sentiment: {emoji} {sentiment_category}</h3>
                        <p style='margin:5px 0 0 0;'>Average Score: {avg_score:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show metrics in 3 columns
            sentiment_cols = st.columns(3)

            # Positive sentiment metric
            with sentiment_cols[0]:
                st.markdown("##### Positive")
                st.markdown(
                    f"<div style='text-align:center; font-size:32px; font-weight:bold; color:#4CAF50;'>{pos_count}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(pos_pct / 100, text=f"{pos_pct:.1f}%")

            # Neutral sentiment with gray gauge
            with sentiment_cols[1]:
                st.markdown("##### Neutral")
                st.markdown(
                    f"<div style='text-align:center; font-size:32px; font-weight:bold; color:#9E9E9E;'>{neu_count}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(neu_pct / 100, text=f"{neu_pct:.1f}%")

            # Negative sentiment with red gauge
            with sentiment_cols[2]:
                st.markdown("##### Negative")
                st.markdown(
                    f"<div style='text-align:center; font-size:32px; font-weight:bold; color:#F44336;'>{neg_count}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(neg_pct / 100, text=f"{neg_pct:.1f}%")

    def display_top_posts(self, analyzed_posts: List[Dict[str, Any]], num_posts: int = 3):
        """
        Display the top posts with their sentiment information.

        This function creates expandable sections for each post,
        showing the title, sentiment, and content.

        Args:
            analyzed_posts: List of posts with sentiment analysis
            num_posts: Number of posts to display
        """
        # Sort posts by Reddit score (most popular first)
        sorted_posts: List[Dict[str, Any]] = sorted(
            analyzed_posts, key=lambda x: x.get("score", 0), reverse=True
        )

        # Display each post in an expander
        for post in sorted_posts[:num_posts]:
            # Get post title and score
            title = post.get("title", "No title")
            author = post.get("author", "Unknown")
            score = post.get("score", 0)

            # Get sentiment information if available
            if "sentiment" in post and "combined" in post["sentiment"]:
                sentiment = post["sentiment"]["combined"]["sentiment"]
                sentiment_score = post["sentiment"]["combined"]["score"]

                # Choose sentiment icon based on sentiment
                if sentiment == "positive":
                    icon = "🟢"  # Green circle for positive
                    color = "green"
                elif sentiment == "negative":
                    icon = "🔴"  # Red circle for negative
                    color = "red"
                else:
                    icon = "⚪"  # White circle for neutral
                    color = "gray"

                sentiment_text = f"{icon} {sentiment.capitalize()} ({sentiment_score:.2f})"
            else:
                sentiment_text = "Sentiment not available"

            # Create expandable section for this post
            with st.expander(f"{title} | Vote: {score} | by /u/{author}"):
                # Show sentiment
                st.write(f"**Sentiment:** {sentiment_text}")

                # Show post content
                selftext = post.get("selftext", "")
                if len(selftext) > 300:
                    st.write(selftext[:300] + "...")
                else:
                    st.write(selftext)
                
                # Show AI-generated sentiment explanation if available
                if "llm_explanation" in post and post["llm_explanation"]:
                    st.markdown(f"**AI Sentiment Explanation:** {post['llm_explanation']}")
                # Show AI-generated summary if available
                if "summary" in post and post["summary"]:
                    st.markdown(f"**AI Summary:** _{post['summary']}_")

    def display_comment_sentiment(self, analyzed_comments: List[Dict[str, Any]]):
        """
        Display sentiment statistics for comments.

        This function calculates and displays metrics about the sentiment
        distribution in the analyzed comments, separately from posts.

        Args:
            analyzed_comments: List of comments with sentiment analysis
        """
        if not analyzed_comments:
            return

        # Extract sentiment information from comments
        sentiments: List[str] = [
            comment["sentiment"]["combined"]["sentiment"]
            for comment in analyzed_comments
            if "sentiment" in comment and "combined" in comment["sentiment"]
        ]

        # Count occurrences of each sentiment type
        pos_count = sentiments.count("positive")
        neu_count = sentiments.count("neutral")
        neg_count = sentiments.count("negative")
        total = len(sentiments)

        # Display metrics only if we have data
        if total > 0:
            # Calculate percentages
            pos_pct: float = (pos_count / total) * 100
            neu_pct: float = (neu_count / total) * 100
            neg_pct: float = (neg_count / total) * 100

            # Calculate average sentiment score for comments
            scores: List[float] = [
                comment["sentiment"]["combined"]["score"]
                for comment in analyzed_comments
                if "sentiment" in comment and "combined" in comment["sentiment"]
            ]

            if scores:
                avg_score: float = sum(scores) / len(scores)

                # Determine overall comment sentiment category
                if avg_score >= SENTIMENT_ANALYSIS["positive_threshold"]:
                    sentiment_category: str = "Positive"
                    color: str = "green"
                    emoji: str = "🟢"
                elif avg_score <= SENTIMENT_ANALYSIS["negative_threshold"]:
                    sentiment_category: str = "Negative"
                    color: str = "red"
                    emoji: str = "🔴"
                else:
                    sentiment_category: str = "Neutral"
                    color: str = "gray"
                    emoji: str = "⚪"

                # Display overall comment sentiment with colored text
                st.markdown(
                    f"""
                    <div style='
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 5px solid {color};
                        margin: 10px 0;'>
                        <h3 style='margin:0;'>Overall Comment Sentiment: {emoji} {sentiment_category}</h3>
                        <p style='margin:5px 0 0 0;'>Average Score: {avg_score:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show metrics in 3 columns
            comment_cols = st.columns(3)

            # Positive sentiment metric for comments
            with comment_cols[0]:
                st.markdown("##### Positive")
                st.markdown(
                    f"<div style='text-align:center; font-size:32px; font-weight:bold; color:#4CAF50;'>{pos_count}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(pos_pct / 100, text=f"{pos_pct:.1f}%")

            # Neutral sentiment metric for comments
            with comment_cols[1]:
                st.markdown("##### Neutral")
                st.markdown(
                    f"<div style='text-align:center; font-size:32px; font-weight:bold; color:#9E9E9E;'>{neu_count}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(neu_pct / 100, text=f"{neu_pct:.1f}%")

            # Negative sentiment metric for comments
            with comment_cols[2]:
                st.markdown("##### Negative")
                st.markdown(
                    f"<div style='text-align:center; font-size:32px; font-weight:bold; color:#F44336;'>{neg_count}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(neg_pct / 100, text=f"{neg_pct:.1f}%")

    def display_top_comments(self, analyzed_comments: List[Dict[str, Any]], num_comments: int = 3):
        """
        Display the top comments with their sentiment information.

        This function creates expandable sections for each comment,
        showing the author, score, sentiment, and content.

        Args:
            analyzed_comments: List of comments with sentiment analysis
            num_comments: Number of comments to display
        """
        # Sort comments by score
        sorted_comments = sorted(
            analyzed_comments, key=lambda x: x.get("score", 0), reverse=True
        )

        # Display top comments
        for i, comment in enumerate(sorted_comments[:num_comments]):
            body = comment.get("body", "No content")
            author = comment.get("author", "Unknown")
            score = comment.get("score", 0)

            # Get sentiment
            if "sentiment" in comment and "combined" in comment["sentiment"]:
                sentiment = comment["sentiment"]["combined"]["sentiment"]
                sentiment_score = comment["sentiment"]["combined"]["score"]

                # Add icon based on sentiment
                if sentiment == "positive":
                    icon = "🟢"
                elif sentiment == "negative":
                    icon = "🔴"
                else:
                    icon = "⚪"

                sentiment_text = f"{icon} {sentiment.capitalize()} ({sentiment_score:.2f})"
            else:
                sentiment_text = "Sentiment not available"

            # Create expandable section for this comment
            with st.expander(f"Comment by /u/{author} | Score: {score}"):
                st.write(f"**Sentiment:** {sentiment_text}")

                # Show truncated comment text if too long
                if len(body) > 200:
                    st.write(body[:200] + "...")
                else:
                    st.write(body)

                # Show AI-generated sentiment explanation if available
                if "llm_explanation" in comment and comment["llm_explanation"]:
                    st.markdown(f"**AI Sentiment Explanation:** {comment['llm_explanation']}")
                
                # Show AI-generated summary if available
                if "summary" in comment and comment["summary"]:
                    st.markdown(f"**AI Summary:** _{comment['summary']}_")
    
    def build_overview_tab(self, analyzed_posts, analyzed_comments):
        """
        Build the Overview tab with sentiment distribution and top posts/comments.
        
        Args:
            analyzed_posts: List of posts with sentiment analysis
            analyzed_comments: List of comments with sentiment analysis
        """
        st.header("Sentiment Distribution")

        posts = st.session_state.posts
        comments = st.session_state.comments

        # AI insights
        if LLM["enabled"] and len(posts) >= 5:
            with st.spinner("Generating AI topic analysis..."):
                subreddit_insights = topic_analysis(posts)
                
                if subreddit_insights:
                    st.markdown("""
                    <div style="padding: 1rem; border-left: 4px solid #FF5700; background-color: #FFF8F0; margin-bottom: 1rem;">
                        <h3 style="margin-top: 0;">🤖 AI Topic Analysis</h3>
                        <p style="margin-bottom: 0.5rem; font-style: italic;">
                    """, unsafe_allow_html=True)
                    
                    st.markdown(subreddit_insights)
                    
                    st.markdown("""
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Post and comment stats
        post_count = len(posts)
        comment_count = len(comments)
        time_range = ""
        if post_count > 0:
            timestamps = [
                p.get("created_utc", 0) for p in posts if "created_utc" in p
            ]
            if timestamps:
                oldest = datetime.fromtimestamp(min(timestamps))
                newest = datetime.fromtimestamp(max(timestamps))
                time_delta = newest - oldest
                if time_delta.days > 0:
                    time_range = f"~{time_delta.days} days"
                else:
                    time_range = f"{time_delta.seconds // 3600} hours"
                    
        # Display metrics
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("Posts Analyzed", f"{post_count}", border=True)
        with stats_cols[1]:
            st.metric("Comments Analyzed", f"{comment_count}", border=True)
        with stats_cols[2]:
            st.metric("Time Range", time_range, border=True)

        # Sentiment explanation
        with st.expander("ℹ️ How to interpret sentiment scores"):
            st.markdown(
                f"""
                - **Positive score** (> {SENTIMENT_ANALYSIS["positive_threshold"]}): The text expresses a favorable or optimistic view
                - **Neutral score** ({SENTIMENT_ANALYSIS["negative_threshold"]} to {SENTIMENT_ANALYSIS["positive_threshold"]}): The text is factual or balanced
                - **Negative score** (< {SENTIMENT_ANALYSIS["negative_threshold"]}): The text expresses a critical or pessimistic view
                
                The analysis combines VADER and TextBlob for more accurate results.
                """
            )

        # Posts section
        st.subheader("Posts", divider=True)

        # Create two columns for layout
        col1, col2 = st.columns(2)

        # Sentiment distribution chart
        fig = plot_sentiment_distribution(
            analyzed_posts,
            figsize=VISUALIZATION["chart_defaults"]["figsize_medium"],
        )
        st.pyplot(fig)

        with col1:
            # Display sentiment statistics
            self.display_sentiment_stats(analyzed_posts)
        with col2:
            # Show top posts
            st.subheader("Top Posts")
            self.display_top_posts(analyzed_posts)

        # Display comment sentiment if available
        if analyzed_comments:
            # Create a separator between post and comment analysis
            st.markdown(
                """
                        
                        """
            )
            st.subheader("Comments", divider=True)

            # Show comment sentiment distribution chart if we have enough comments
            if len(analyzed_comments) >= 5:
                comment_col1, comment_col2 = st.columns(2)

                # Create comment sentiment distribution chart
                comment_fig = plot_sentiment_distribution(
                    analyzed_comments,
                    title="Comment Sentiment Distribution",
                    figsize=VISUALIZATION["chart_defaults"]["figsize_medium"],
                )
                st.pyplot(comment_fig)

                with comment_col1:
                    # Show comment sentiment statistics
                    self.display_comment_sentiment(analyzed_comments)

                with comment_col2:
                    # Show top comments
                    st.subheader("Top Comments")
                    self.display_top_comments(analyzed_comments)
    
    def build_trends_tab(self, analyzed_posts, analyzed_comments):
        """
        Build the Trends tab with sentiment over time visualizations.
        
        Args:
            analyzed_posts: List of posts with sentiment analysis
            analyzed_comments: List of comments with sentiment analysis
        """
        st.header("Sentiment Trends Over Time")

        # Add a cleaner info box
        st.markdown(
            f"""
            <div style="padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h4 style="margin-top: 0;">How to Read These Charts</h4>
                <ul>
                    <li><strong>Top chart:</strong> Average sentiment score (-1 to 1) over time</li>
                    <li><strong>Bottom chart:</strong> Count of posts by sentiment type</li>
                    <li>Scores above {SENTIMENT_ANALYSIS["positive_threshold"]} are positive,
                        below {SENTIMENT_ANALYSIS["negative_threshold"]} are negative, and in between are neutral</li>
                </ul>
                <p>These trends can reveal how community sentiment shifts over time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Create time series chart using visualization.py
        fig = plot_sentiment_over_time(analyzed_posts)
        st.pyplot(fig)

        # Show comment sentiment trends if available and we have enough data
        if analyzed_comments and len(analyzed_comments) >= 10:
            st.markdown("---")
            st.header("Comment Sentiment Trends")

            # Create time series chart for comments
            comment_trend_fig = plot_sentiment_over_time(analyzed_comments)
            st.pyplot(comment_trend_fig)
    
    def build_word_analysis_tab(self, analyzed_posts, analyzed_comments):
        """
        Build the Word Analysis tab with word clouds and top terms.
        
        Args:
            analyzed_posts: List of posts with sentiment analysis
            analyzed_comments: List of comments with sentiment analysis
        """
        st.header("Word Analysis")
        posts_to_display = analyzed_posts
        comments_to_display = analyzed_comments

        with st.form(key="word_analysis_form"):
        # Create two columns for layout
            col1, col2 = st.columns(2)
            with col1:
                # Add sentiment filter dropdown
                sentiment = st.selectbox(
                    "Filter by sentiment",
                    options=["All", "Positive", "Neutral", "Negative"],
                    index=["All", "Positive", "Neutral", "Negative"].index(
                        st.session_state.sentiment_filter
                    ),
                    help="Show words only from posts with selected sentiment",
                    key="sentiment_filter_select"
                )

            with col2:
                # Slider for number of terms to show
                top_n = st.slider(
                    "Number of terms",
                    min_value=5,
                    max_value=20,
                    value=st.session_state.term_count,
                    help="Number of most frequent terms to display",
                    key="term_count_slider"
                )

            # Submit button for the form
            filter_submitted = st.form_submit_button("Apply Filters")
            
            # Update session state when form is submitted
            if filter_submitted:
                st.session_state.sentiment_filter = sentiment
                st.session_state.term_count = top_n

        # Get current values from session state
        current_sentiment = st.session_state.sentiment_filter
        current_term_count = st.session_state.term_count

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            # Word cloud for posts
            st.subheader("Post Word Cloud")

            # Convert selection to filter value
            # None means no filter (all sentiments)
            filter_value = None if current_sentiment == "All" else current_sentiment.lower()

            # Generate word cloud using visualization.py
            cloud_fig = generate_wordcloud(
                posts_to_display, sentiment_filter=filter_value
            )
            st.pyplot(cloud_fig)

        with col2:
            # Top terms chart for posts
            st.subheader("Top Terms in Posts")

            # Generate terms chart using visualization.py
            terms_fig = create_top_terms_chart(
                posts_to_display, top_n=current_term_count, sentiment_filter=filter_value
            )
            st.pyplot(terms_fig)

        # Show comment word analysis if available
        if comments_to_display and len(comments_to_display) >= 5:

            comment_col1, comment_col2 = st.columns(2)

            with comment_col1:
                # Word cloud for comments
                st.subheader("Comment Word Cloud")

                # Generate word cloud for comments
                comment_cloud_fig = generate_wordcloud(
                    comments_to_display, sentiment_filter=filter_value
                )
                st.pyplot(comment_cloud_fig)

            with comment_col2:
                # Top terms in comments
                st.subheader("Top Terms in Comments")

                # Generate top terms chart for comments
                comment_terms_fig = create_top_terms_chart(
                    comments_to_display, top_n=current_term_count, sentiment_filter=filter_value
                )
                st.pyplot(comment_terms_fig)
    
    def build_data_tab(self, analyzed_posts, analyzed_comments):
        """
        Build the Data tab with tables of posts and comments.
        
        Args:
            analyzed_posts: List of posts with sentiment analysis
            analyzed_comments: List of comments with sentiment analysis
        """
        st.header("Data Tables")

        # Create tabs for post and comment data
        data_tabs = st.tabs(["Posts", "Comments"])

        # Posts data tab
        with data_tabs[0]:
            st.subheader("Post Data")

            # Convert post data to DataFrame for display
            df = pd.DataFrame(
                [
                    {
                        "title": post.get("title", ""),
                        "author": post.get("author", "No author"),
                        "selftext": post.get("selftext", ""),
                        "subreddit": post.get("subreddit", ""),
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "created": datetime.fromtimestamp(
                            post.get("created_utc", 0)
                        ).strftime("%Y-%m-%d %H:%M"),
                        "post_id": post.get("id", ""),
                        "sentiment": post.get("sentiment", {})
                        .get("combined", {})
                        .get("sentiment", ""),
                        "sentiment_score": post.get("sentiment", {})
                        .get("combined", {})
                        .get("score", 0),
                    }
                    for post in analyzed_posts
                ]
            )

            # Display the DataFrame as a table
            st.dataframe(df)

            # Add download button for CSV export
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Post Data",
                csv,
                f"reddit_posts_{st.session_state.subreddit}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                on_click=None,
            )

        # Comments data tab
        with data_tabs[1]:
            st.subheader("Comment Data")

            if analyzed_comments:
                # Convert comments to DataFrame
                comments_df = pd.DataFrame(
                    [
                        {
                            "body": comment.get("body", ""),
                            "author": comment.get("author", "No author"),
                            "subreddit": comment.get("subreddit", ""),
                            "score": comment.get("score", 0),
                            "created": (
                                datetime.fromtimestamp(
                                    comment.get("created_utc", 0)
                                ).strftime("%Y-%m-%d %H:%M")
                                if "created_utc" in comment
                                else ""
                            ),
                            "post_id": comment.get("post_id", ""),
                            "sentiment": comment.get("sentiment", {})
                            .get("combined", {})
                            .get("sentiment", ""),
                            "sentiment_score": comment.get("sentiment", {})
                            .get("combined", {})
                            .get("score", 0),
                        }
                        for comment in analyzed_comments
                    ]
                )

                # Display comments table
                st.dataframe(comments_df)

                # Add download button for comments CSV
                comments_csv = comments_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Comment Data",
                    comments_csv,
                    f"reddit_comments_{st.session_state.subreddit}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    on_click=None,
                )
            else:
                st.info(
                    "No comment data available. Enable 'Include comments' in the sidebar to fetch comments."
                )

    def build_comparison_tab(self, main_subreddit, comparison_subreddit):
        """
        Build the Comparison tab with subreddit comparison visualizations.
        
        Args:
            main_subreddit: Primary subreddit name
            comparison_subreddit: Secondary subreddit to compare with
        """
        st.header(f"Subreddit Comparison")
        
        # Get data for both subreddits
        main_data = st.session_state.analyzed_subreddits.get(main_subreddit, {}).get('posts', [])
        comp_data = st.session_state.analyzed_subreddits.get(comparison_subreddit, {}).get('posts', [])
        
        if not main_data or not comp_data:
            st.warning("Missing data for one or both subreddits.")
            return
        
        # Display post counts for transparency
        st.info(f"Comparing {len(main_data)} posts from r/{main_subreddit} with {len(comp_data)} posts from r/{comparison_subreddit}")
        
        # Create a dictionary of subreddit data for the comparison function
        comparison_data = {
            main_subreddit: main_data,
            comparison_subreddit: comp_data
        }
        
        # Use the visualization function to create the comparison chart
        comparison_fig = plot_subreddit_comparison(comparison_data)
        st.pyplot(comparison_fig)
        
        # Add explanation
        st.markdown(f"""
        ### How to Read This Chart
        
        **Top Chart:**
        - Each bar shows the percentage breakdown of positive, neutral, and negative posts
        - Higher positive percentages (green) indicate more positive sentiment
        
        **Bottom Chart:**
        - Shows the average sentiment score for each subreddit
        - Scores above {SENTIMENT_ANALYSIS["positive_threshold"]} are positive, below {SENTIMENT_ANALYSIS["negative_threshold"]} are negative
        """)
        
        # Add comments comparison if available
        main_comments = st.session_state.analyzed_subreddits.get(main_subreddit, {}).get('comments', [])
        comp_comments = st.session_state.analyzed_subreddits.get(comparison_subreddit, {}).get('comments', [])
        
        if main_comments and comp_comments:
            st.markdown("---")
            st.subheader("Comment Sentiment Comparison")
            st.info(f"Comparing {len(main_comments)} comments from r/{main_subreddit} with {len(comp_comments)} comments from r/{comparison_subreddit}")
            
            comment_comparison_data = {
                main_subreddit: main_comments,
                comparison_subreddit: comp_comments
            }
            
            comment_comparison_fig = plot_subreddit_comparison(
                comment_comparison_data, 
                figsize=(12, 8)
            )
            st.pyplot(comment_comparison_fig)
    
    def display_welcome_screen(self):
        """
        Display the welcome screen when no analysis has been performed.
        
        This shows getting started instructions and information about the project.
        """
        st.markdown(
            """      
            To get started:
            
            1. Select or enter a subreddit in the sidebar
            2. Optionally, check 'Filter for keywords' and enter keywords to filter posts
            3. Check 'Include comments' if you want to analyze comments too
            4. Set the number of posts (and comments) to analyze
            5. Click "Analyze Subreddit"
            
            The app will fetch data, analyze sentiment, and show visualizations of the results.
            """
        )

        st.markdown("---")  # Add a separator
        st.markdown("## About this Project")

        st.markdown(
            """
            The **Reddit Sentiment Analyzer** is a tool that analyzes the sentiment of discussions on Reddit subreddits. 
            It provides insights into whether communities are generally positive, negative, or neutral about topics.
            
            ### Features
            - Real-time data from Reddit API
            - Advanced sentiment analysis using VADER and TextBlob
            - Word clouds and term frequency analysis
            - Time-based trend visualization
            - Data exporting capabilities
            """
        )

        st.markdown(
            """
            ### Technical Stack
            - **Backend**: Python, PRAW, NLTK, VADER, TextBlob
            - **Data Storage**: MongoDB
            - **Visualization**: Matplotlib, WordCloud
            - **Web Interface**: Streamlit
            - **Deployment**: Docker
            """
        )
    
    def display_analysis_results(self):
        """
        Display analysis results when data has been analyzed.
        
        This creates tabs for different views of the analyzed data.
        """
        # Style the tabs
        st.markdown(
            """
        <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 15px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 15px;
                border-radius: 4px;
            }
            .stTabs [aria-selected="true"] {
                background-color: rgba(255, 69, 0, 0.1);
                border-bottom-color: rgb(255, 69, 0);
            }
        </style>""",
            unsafe_allow_html=True,
        )
        
        # Determine if we should show comparison tab
        tab_names = ["📊 Overview", "📈 Trends", "🔤 Word Analysis", "📋 Data"]
        main_subreddit = st.session_state.subreddit
        comp_subreddit = st.session_state.comparison_subreddit
        
        # Add comparison tab if comparison mode is enabled and we have data for both subreddits
        has_comparison_data = (
            st.session_state.comparison_enabled and 
            comp_subreddit and
            comp_subreddit in st.session_state.analyzed_subreddits
        )
        
        if has_comparison_data:
            tab_names.append("⚖️ Comparison")
        
        # Get analysis data from session state
        analyzed_posts = st.session_state.analyzed_posts
        analyzed_comments = st.session_state.analyzed_comments

        # Create tabs
        tabs = st.tabs(tab_names)
        
        # Build each tab's content
        with tabs[0]:
            self.build_overview_tab(analyzed_posts, analyzed_comments)
            
        with tabs[1]:
            self.build_trends_tab(analyzed_posts, analyzed_comments)
            
        with tabs[2]:
            self.build_word_analysis_tab(analyzed_posts, analyzed_comments)
            
        with tabs[3]:
            self.build_data_tab(analyzed_posts, analyzed_comments)
        
        # Add comparison tab if needed
        if has_comparison_data:
            with tabs[4]:  # The fifth tab (index 4)
                self.build_comparison_tab(main_subreddit, comp_subreddit)
            
        # Show status at the bottom
        st.markdown("---")
        status_container = st.empty()
        status_container.success(
            f"Analyzed {len(analyzed_posts)} posts from r/{st.session_state.subreddit}"
        )
        
        # Show detailed processing status in an expander
        with st.expander("Process details", expanded=False):
            st.text(st.session_state.fetch_status + st.session_state.analysis_status)
    
    def run(self):
        """
        Run the Streamlit application.
        
        This is the main entry point that controls the application flow.
        """
        # Create the header
        self.build_header()
        
        # Build the sidebar and get user inputs
        self.build_sidebar()
        
        # Display either results or welcome screen
        if st.session_state.has_analyzed and st.session_state.analyzed_posts:
            self.display_analysis_results()
        else:
            self.display_welcome_screen()


def main():
    """
    Main entry point for the application.
    
    Initializes and runs the RedditSentimentApp.
    """
    app = RedditSentimentApp()
    app.run()


if __name__ == "__main__":
    main()
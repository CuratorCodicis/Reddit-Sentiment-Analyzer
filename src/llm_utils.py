"""
LLM Utilities Module for Reddit Sentiment Analyzer

This module provides functions for using a local LLM to enhance sentiment analysis:
- Summarize Reddit posts and comments
- Explain sentiment scores
- Identify topics across multiple posts/comments
- Generate insights about community sentiment trends

The module uses the Phi-2 model via llama-cpp-python for efficient inference.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union

# Import the llama_cpp library for model inference
try:
    from llama_cpp import Llama
    LLAMA_IMPORT_ERROR = None
except ImportError as e:
    LLAMA_IMPORT_ERROR = str(e)
    logging.warning(f"Failed to import llama_cpp: {e}")
    logging.warning("LLM functionality will be disabled")

from config import LLM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Global variable to hold the model instance
_model = None

def load_model() -> Optional[Any]:
    """
    Load the LLM model based on configuration settings.
    
    This function initializes the model with appropriate parameters
    for memory usage, GPU acceleration, and context length.
    
    Returns:
        Optional[Any]: The loaded model or None if loading fails
    """
    global _model
    
    # If model is already loaded, return it
    if _model is not None:
        return _model
    
    # If llama_cpp failed to import, we can't load the model
    if LLAMA_IMPORT_ERROR:
        logging.warning(f"Cannot load model due to import error: {LLAMA_IMPORT_ERROR}")
        return None
    
    # Check if model file exists
    model_path = LLM["model_path"]
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        logging.error("Please download the model and place it in the correct path.")
        logging.error("See README.md for instructions.")
        return None
    
    try:
        # Configure GPU usage based on settings
        n_gpu_layers = 0
        if LLM["use_gpu"]:
            n_gpu_layers = LLM["gpu_layers"]
            logging.info(f"GPU acceleration enabled, using {n_gpu_layers if n_gpu_layers else 'all available'} layers")
        
        # Load the model with appropriate settings
        logging.info(f"Loading {model_path}...")
        _model = Llama(
            model_path=model_path,
            n_ctx=LLM["context_length"],  # Context window size
            n_batch=512,  # Batch size for prompt processing
            n_gpu_layers=n_gpu_layers,  # Number of layers to offload to GPU
            use_mlock=True,  # Lock memory to prevent swapping
            verbose=False,  # Reduce console output
        )
        
        logging.info(f"Model successfully loaded with context length {LLM['context_length']}")
        return _model
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None
    
def generate_response(prompt: str, max_tokens: int = None, temperature: float = None) -> str:
    """
    Generate a response from the LLM model for the given prompt.
    
    Args:
        prompt (str): The input prompt to send to the model
        max_tokens (int, optional): Maximum number of tokens to generate
        temperature (float, optional): Temperature for sampling (higher = more random)
        
    Returns:
        str: The generated response text or an error message
    """
    # Use default values from config if not specified
    if max_tokens is None:
        max_tokens = LLM["max_tokens"]
    if temperature is None:
        temperature = LLM["temperature"]
    
    # Try to load the model if it's not loaded yet
    model = load_model()
    if model is None:
        return "[LLM ERROR: Model not available]"
    
    try:
        # Use a timeout to prevent hanging
        start_time = time.time()
        
        # Generate response with the model
        response = model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["##", "USER:", "ASSISTANT:", "###", "<<<", ">>>", "\n\n", "STOP"],  # Stop sequences to end generation
            echo=False,  # Don't include the prompt in the response
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        logging.info(f"Generated response in {generation_time:.2f} seconds")
        
        # Extract the text from the response
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["text"].strip()
        else:
            logging.warning("Empty response from model")
            return "[No response generated]"
            
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"[LLM ERROR: {str(e)}]"
    
def construct_summary_prompt(text: str) -> str:
    """
    Construct a prompt for summarizing text.
    
    Args:
        text (str): The text to summarize
        
    Returns:
        str: A well-formatted prompt for the LLM
    """
    return f"""You are a text summarizer. Provide ONLY a brief, 1-sentence summary of this Reddit post. 
    Focus only on the main topic or question. Do not react to or answeer anything from the Reddit post. Be direct and concise.
    Do NOT include any quotation marks, code fences, or extra commentary—just the sentence.

    When you are finished with your reponse write STOP so we can end the reponse.
    
    REDDIT POST:
    {text}

    ONE SENTENCE SUMMARY:"""

def summarize_text(text: str) -> str:
    """
    Summarize a piece of text (post or comment) in a concise way.
    
    Args:
        text (str): The text to summarize
        
    Returns:
        str: A concise summary of the text
    """
    if not text or len(text.strip()) < 5:
        return text  # Text is too short to summarize
    
    # Check if LLM is enabled in config
    if not LLM["enabled"]:
        return ""
    
    # Construct the prompt and generate the summary
    prompt = construct_summary_prompt(text)
    summary = generate_response(prompt, max_tokens=60, temperature=0.3)
    
    return summary

def construct_sentiment_explanation_prompt(text: str, sentiment: str, score: float) -> str:
    """
    Construct a prompt for explaining sentiment analysis results.
    
    Args:
        text (str): The text whose sentiment was analyzed
        sentiment (str): The sentiment label (positive, negative, neutral)
        score (float): The sentiment score (-1 to 1)
        
    Returns:
        str: A well-formatted prompt for the LLM
    """
    return f"""Explain in ONE SENTENCE why this Reddit content received a sentiment score of {score:.2f} ({sentiment.upper()}).
    Focus only on specific words/phrases that influenced this sentiment classification.
    Do NOT include any quotation marks, code fences, or extra commentary—just the sentence.

    When you are finished with your reponse write STOP so we can end the reponse.

    CONTENT:
    {text}

    ONE SENTENCE EXPLANATION:"""

def explain_sentiment(text: str, sentiment: str, score: float) -> str:
    """
    Generate an explanation for why a specific sentiment score was assigned.
    
    Args:
        text (str): The text whose sentiment was analyzed
        sentiment (str): The sentiment label (positive, negative, neutral)
        score (float): The sentiment score (-1 to 1)
        
    Returns:
        str: An explanation of the sentiment classification
    """
    if not text or len(text.strip()) < 5:
        return f"[Text too short for meaningful explanation]"
    
    # Check if LLM is enabled in config
    if not LLM["enabled"]:
        return ""
    
    # Construct the prompt and generate the explanation
    prompt = construct_sentiment_explanation_prompt(text, sentiment, score)
    explanation = generate_response(prompt, max_tokens=60, temperature=0.3)
    
    return explanation

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single Reddit item (post or comment) with the LLM.
    
    This function adds summary and sentiment explanation to the item.
    
    Args:
        item (Dict[str, Any]): A Reddit post or comment dictionary
        
    Returns:
        Dict[str, Any]: The same item with added LLM-generated fields
    """
    # Skip processing if LLM is disabled
    if not LLM["enabled"]:
        return item
    
    # Create a copy to avoid modifying the original
    result = item.copy()
    
    # Determine if it's a post or comment
    if "title" in item:
        # It's a post
        # For posts, summarize the selftext if it exists and is substantial
        if item.get("selftext") and len(item["selftext"].strip()) > 5:
            # Combine title and selftext for more context
            full_text = f"{item['title']}\n\n{item['selftext']}"
            result["summary"] = summarize_text(full_text)
        else:
            # Just the title if there's no significant selftext
            result["summary"] = summarize_text(item["title"])
            
    elif "body" in item:
        # It's a comment
        result["summary"] = summarize_text(item["body"])
    
    # Add sentiment explanation if sentiment analysis is available
    if "sentiment" in item and "combined" in item["sentiment"]:
        sentiment = item["sentiment"]["combined"]["sentiment"]
        score = item["sentiment"]["combined"]["score"]
        
        # Determine which text to explain
        if "title" in item:
            # For posts, use both title and selftext if available
            if item.get("selftext"):
                text_to_explain = f"{item['title']}\n\n{item['selftext']}"
            else:
                text_to_explain = item["title"]
        else:
            # For comments, use the body
            text_to_explain = item["body"]
        
        # Generate the explanation
        result["llm_explanation"] = explain_sentiment(text_to_explain, sentiment, score)
    
    return result

def process_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of Reddit items (posts or comments) with the LLM.
    
    Args:
        items (List[Dict[str, Any]]): List of Reddit posts or comments
        
    Returns:
        List[Dict[str, Any]]: The same items with added LLM-generated fields
    """
    # Skip processing if LLM is disabled
    if not LLM["enabled"]:
        return items
    
    processed_items = []
    total_items = len(items)
    
    logging.info(f"Processing {total_items} items with LLM...")
    
    for i, item in enumerate(items):
        # Process each item
        processed_item = process_item(item)
        processed_items.append(processed_item)
        
        # Log progress for every 10 items or at the end
        if (i + 1) % 10 == 0 or i == total_items - 1:
            logging.info(f"Processed {i + 1}/{total_items} items")
    
    return processed_items

def construct_topic_analysis_prompt(items: List[str]) -> str:
    """
    Construct a prompt for topic analysis from multiple summaries.
    
    Args:
        summaries (List[str]): List of summarized Reddit content
        
    Returns:
        str: A well-formatted prompt for the LLM
    """
    # Build one bullet per item: "- [POSITIVE +0.32] summary..."
    bullets = []
    for it in items:
        # grab summary
        summary = it.get("summary", "").replace("\n", " ").strip()
        # grab combined sentiment info
        combined = it.get("sentiment", {}).get("combined", {})
        label = combined.get("sentiment", "neutral").upper()
        score = combined.get("score", 0.0)
        bullets.append(f"- [{label} {score:+.2f}] {summary}")

    bullet_block = "\n".join(bullets)
    
    return f"""Based on the following summaries of Reddit posts/comments (each tagged with sentiment and score), please:
    1. Identify 1-3 main topics or themes being discussed.
    2. Note any significant disagreements or contrasting viewpoints.
    3. Try to find an explanation for the general sentiment given the context.
    Write 3 sentences max. discussing your findings. ONLY WRITE A CONTINUOUS TEXT! Do NOT use bulletpoints. Do NOT use lists. Do NOT use enumerations.
    Do NOT include any quotation marks, code fences, or extra commentary—just the sentences. do NOT acknowledge the existence of summaries of reddits items and talk like you read the items themselves.
    Do NOT reference specifix scores but rather overall sentiment descriptions.
    Remember that this analysis is the centerpiece of the Sentiment Distribution and thus be well fomulated and representative of the whole project.

    When you are finished with your reponse write STOP so we can end the reponse.

    SUMMARIES:{bullet_block}

    ANALYSIS:"""

def topic_analysis(items: List[Dict[str, Any]]) -> str:
    """
    Analyze topics and overall sentiment across multiple Reddit items.
    
    This function identifies common themes, sentiment patterns, and
    interesting insights across a collection of posts or comments.
    
    Args:
        items (List[Dict[str, Any]]): List of Reddit posts or comments
        
    Returns:
        str: A comprehensive analysis of topics and sentiment
    """
    # Skip analysis if LLM is disabled or there are no items
    if not LLM["enabled"] or not items:
        return ""
    
    # First, ensure items have summaries
    for item in items:
        if "summary" not in item:
            # Process the item to generate a summary
            processed_item = process_item(item)
            item["summary"] = processed_item.get("summary", "")
    
    # Extract summaries and filter out empty ones
    items_to_analyze = [item for item in items if item.get("summary")]
    
    # If we have no valid summaries, return empty string
    if not items_to_analyze:
        return ""
    
    # Construct the prompt and generate the analysis
    prompt = construct_topic_analysis_prompt(items_to_analyze)
    analysis = generate_response(prompt, max_tokens=200, temperature=0.5)
    
    return analysis

# Initialize the model when the module is imported
if LLM["enabled"]:
    try:
        _ = load_model()
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        logging.warning("LLM functionality will be disabled")
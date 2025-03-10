# Reddit Sentiment Analyzer

The **Reddit Sentiment Analyzer** is a Python-based tool that analyzes sentiment trends in Reddit discussions. Users can select any subreddit and analyze how positive, negative, or neutral the community's discussions are. With visualization tools and real-time data collection, this application provides insights into Reddit communities' emotional patterns and trending topics.
<br/>
<br/>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d017d123-6dae-4edc-9119-40e6e8a5541e" width=75%>
</p>


---

## 📊 Features & Workflow

- **Real-Time Reddit Data Collection:** Fetch posts and (optionally) comments from any subreddit using PRAW (Python Reddit API Wrapper), preprocess text, and store structured data in MongoDB.
  
- **Multi-Method Sentiment Analysis:** Classify text as positive, neutral, or negative using both VADER and TextBlob for better accuracy.

- **Interactive Data Visualization:** Explore sentiment trends with:
  - Sentiment distribution pie charts
  - Time-series analysis showing sentiment evolution
  - Word clouds by sentiment type
  - Top terms frequency analysis
  
- **Customizable Analysis:**
  - Filter posts by keywords
  - Adjust number of posts and comments to analyze
  - Compare sentiment across different subreddits
  
- **Data Persistence:** MongoDB-based caching to optimize performance and minimize redundant API calls.

- **User-Friendly Interface:** Clean Streamlit-based web application for easy navigation and analysis.

- **Containerized Deployment:** Easily deploy and run with Docker.

---

## 📦 Installation & Setup

### **1️⃣ Run with Docker**

Make sure you have Docker and Docker Compose installed, then:

```bash
# Clone the repository
git clone https://github.com/your-username/reddit-sentiment-analyzer.git
cd reddit-sentiment-analyzer

# Create a .env file with your Reddit API credentials
# (see .env.example for required variables)

# Build and start the containers
docker compose build
docker compose up
```

The application will be available at **http://localhost:8501**.

To stop and remove containers:

```bash
docker compose down
```

### **2️⃣ Manual Installation**

If you prefer a manual setup:

#### **Prerequisites**

- Python 3.11+
- MongoDB (running locally on default port or configured via .env)
- (Optional) NLTK data for VADER—will be auto-downloaded if missing

#### **Step-by-Step Setup**

```bash
# Clone the repository
git clone https://github.com/your-username/reddit-sentiment-analyzer.git
cd reddit-sentiment-analyzer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file based on .env.example with your Reddit API credentials
```

#### **Running the Application**

```bash
# Start the application
streamlit run src/app.py
```

The Streamlit web interface will open at **http://localhost:8501**.

---

## 🖥️ Getting Reddit API Credentials

To use this application, you'll need Reddit API credentials:

1. **Create a Reddit Account** if you don't have one
2. **Go to [Reddit's App Preferences](https://www.reddit.com/prefs/apps)**
3. **Click "Create App" or "Create Another App"**
4. **Fill in the details:**
   - Name: Reddit Sentiment Analyzer (or your preferred name)
   - App type: Script
   - Description: Brief description of the app
   - About URL: Your GitHub repository or personal website (optional)
   - Redirect URI: http://localhost:8501
5. **Click "Create app"**
6. **Note your credentials:**
   - Client ID: The string under "personal use script"
   - Client Secret: The string listed as "secret"
7. **Add these to your .env file as shown in .env.example**

---

## 🎮 Usage Guide

1. **Select a Subreddit:**
   - Enter a custom subreddit name or choose from the samples
   - Optionally enable keyword filtering to focus on specific topics

2. **Configure Analysis Settings:**
   - Set the number of posts to analyze (10-1000)
   - Choose whether to include comments
   - Enable subreddit comparison (optional)

3. **View Analysis Results:**
   - **Overview Tab:** Sentiment distribution and top posts
   - **Trends Tab:** Time-series analysis of sentiment changes
   - **Word Analysis Tab:** Word clouds and top terms (filterable by sentiment)
   - **Data Tab:** Exportable tables of posts and comments with sentiment scores
   - **Comparison Tab:** Side-by-side comparison of two subreddits (if enabled)

4. **Export Data:**
   - Download post and comment data as CSV files if of interest

---

## 🧩 Project Structure

```
reddit-sentiment-analyzer/
├── src/
│   ├── app.py                # Main Streamlit application
│   ├── config.py             # Configuration and environment settings
│   ├── data_preprocessing.py # Text cleaning and preprocessing
│   ├── database.py           # MongoDB connection and operations
│   ├── reddit_api.py         # Reddit API interaction using PRAW
│   ├── sentiment_analysis.py # VADER and TextBlob sentiment analysis
│   └── visualization.py      # Charts and visualization functions
├── .dockerignore             # Files to exclude from Docker build
├── .env.example              # Template for environment variables
├── .gitignore                # Git ignore patterns
├── Dockerfile                # Docker image configuration
├── README.md                 # Project documentation
├── docker-compose.yml        # Docker configuration
├── mongo-init.js             # MongoDB initialization script
└── requirements.txt          # Python dependencies
```

---

## 🛠️ Tech Stack

| Category | Technology/Tool |
|----------|-----------------|
| **Language** | Python 3.11 |
| **Reddit API** | PRAW (Python Reddit API Wrapper) |
| **Database** | MongoDB |
| **NLP & Sentiment Analysis** | VADER, TextBlob, NLTK |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Web Interface** | Streamlit |
| **Deployment** | Docker |

---

## 📷 Example Visualization Screenshots

![Overview](https://github.com/user-attachments/assets/b188ffc4-258b-4b1f-a542-93e8e5365037)

![Trends](https://github.com/user-attachments/assets/1a5c2acb-4549-4cc1-8a78-99e57897a91f)

![Word Analysis](https://github.com/user-attachments/assets/dd4b3c0b-7619-408d-a099-16b417778167)

![Data](https://github.com/user-attachments/assets/ccc94180-eaa6-4b29-b0cc-b0b7a38751cd)

![Comparison](https://github.com/user-attachments/assets/3aae5186-5e3f-46f6-bd34-c82082dd4064)

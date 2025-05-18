import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reddit API credentials — REPLACE these with your values
reddit = praw.Reddit(
    client_id="4UL0BHqGQZhD4bvRw214Mg",
    client_secret=None,
    user_agent="android_vs_iphone_script"
)

reddit.read_only = True  # ✅ tell PRAW we’re just browsing

analyzer = SentimentIntensityAnalyzer()

# Fetch posts from r/technology that mention a keyword
def fetch_posts(keyword, limit=100):
    posts = []
    for submission in reddit.subreddit("technology").search(keyword, limit=limit):
        sentiment = analyzer.polarity_scores(submission.title + " " + submission.selftext)
        posts.append({
            "title": submission.title,
            "text": submission.selftext,
            "score": submission.score,
            "sentiment": sentiment["compound"]
        })
    return pd.DataFrame(posts)

# Run the script
if __name__ == "__main__":
    df_android = fetch_posts("Android phone")
    df_iphone = fetch_posts("iPhone")

    df_android["label"] = "Android"
    df_iphone["label"] = "iPhone"

    combined = pd.concat([df_android, df_iphone])
    combined.to_csv("data/reddit_sentiment.csv", index=False)

    print("Saved sentiment results to data/reddit_sentiment.csv")

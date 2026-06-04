import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("comments.csv")

# Sentiment function
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    else:
        return "Negative"

# Apply sentiment analysis
df["Sentiment"] = df["Comment"].apply(get_sentiment)

print(df)

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

# Graph
plt.bar(sentiment_counts.index, sentiment_counts.values)

plt.title("Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig("output.png")
plt.show()
import pandas as pd
import nltk
nltk.download('punkt')
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt

# Example dataset for TextBlob
texts = [
    "I love Python, it's amazing!",
    "This movie was terrible and boring.",
    "The food was okay, not great but not bad either.",
    "This movie was fantastic and thrilling!",
    "Absolutely terrible customer service.",
    "The product is decent, nothing extraordinary.",
    "I'm so happy with my purchase!",
    "Really frustrating experience, never again."
]

# TextBlob sentiment analysis
for text in texts:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Range: -1 (negative) to +1 (positive)
    
    if sentiment > 0:
        label = "Positive"
    elif sentiment < 0:
        label = "Negative"
    else:
        label = "Neutral"
    
    print(f"Text: {text}\nSentiment Score: {sentiment:.2f} → {label}\n")

# Example dataset for DataFrame
data = {'review': [
    "The product is great and works well!",
    "Worst purchase ever, completely useless.",
    "It's fine, nothing special."
]}
df = pd.DataFrame(data)
df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment'] = df['polarity'].apply(lambda x: "Positive" if x>0 else ("Negative" if x<0 else "Neutral"))
print(df)

# Hugging Face Transformers sentiment analysis
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


texts = [
    "I love using ChatGPT!",
    "I really hate the traffic today.",
    "The weather is just average."
]

results = sentiment_pipeline(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}\nSentiment: {result['label']} (Score: {result['score']:.2f})\n")

# Plot sentiment distribution from TextBlob DataFrame
sentiment_counts = df['sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

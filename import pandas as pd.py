import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence

# Initialize VADER and Flair
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
flair_sentiment = TextClassifier.load('en-sentiment')

# Load your dataset
file_name = 'True.csv'
data = pd.read_csv(file_name)

# TextBlob
def label_sentiment_textblob(sentence):
    sentiment = TextBlob(sentence).sentiment.polarity
    return 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'

# VADER
def label_sentiment_vader(sentence):
    sentiment = sia.polarity_scores(sentence)
    return 'positive' if sentiment['compound'] > 0.05 else 'negative' if sentiment['compound'] < -0.05 else 'neutral'

# Flair
def label_sentiment_flair(sentence):
    sentence = Sentence(sentence)
    flair_sentiment.predict(sentence)
    sentiment = sentence.labels[0]
    return 'positive' if 'POSITIVE' in str(sentiment) else 'negative'

# Apply sentiment analysis
sentences_data = []
for text in data['text']:
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentiment_textblob = label_sentiment_textblob(sentence)
        sentiment_vader = label_sentiment_vader(sentence)
        sentiment_flair = label_sentiment_flair(sentence)
        
        # Majority vote for final sentiment
        sentiments = [sentiment_textblob, sentiment_vader, sentiment_flair]
        final_sentiment = max(set(sentiments), key = sentiments.count)

        sentences_data.append({
            'sentence': sentence,
            'sentiment_textblob': sentiment_textblob,
            'sentiment_vader': sentiment_vader,
            'sentiment_flair': sentiment_flair,
            'final_sentiment': final_sentiment
        })

# Create a new dataframe
sentences_df = pd.DataFrame(sentences_data)

# Save to a new CSV file
sentences_df.to_csv('sentences_sentiment.csv', index=False)

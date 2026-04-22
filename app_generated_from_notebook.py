
import json
import re
import joblib
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
lda = joblib.load('./model/topic_model_lda.pkl')
vectorizer = joblib.load('./model/topic_vectorizer.pkl')
with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
    topic_labels = {int(k): v for k, v in json.load(f).items()}
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)
def predict(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    probs = lda.transform(X)[0]
    topic_id = int(np.argmax(probs))
    sentiment_score = sia.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        sentiment = 'positive'
    elif sentiment_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return {
        'topic': topic_labels.get(topic_id, f'Topic {topic_id + 1}'),
        'topic_probability': float(probs[topic_id]),
        'sentiment': sentiment,
        'sentiment_compound_score': float(sentiment_score)
    }
st.title('Course Evaluation NLP Demo')
st.write('Enter a new comment to predict topic and sentiment.')
user_text = st.text_area('Student comment', 'The labs were practical and useful, but more NLP examples would help.')
if st.button('Predict'):
    out = predict(user_text)
    st.success(f"Predicted topic: {out['topic']} (probability={out['topic_probability']:.3f})")
    st.info(f"Predicted sentiment: {out['sentiment']} (compound={out['sentiment_compound_score']:.3f})")

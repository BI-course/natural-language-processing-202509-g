from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TOPIC_MODEL_PATH = MODEL_DIR / "topic_model_lda.pkl"
VECTORIZER_PATH = MODEL_DIR / "topic_vectorizer.pkl"
TOPIC_META_PATH = MODEL_DIR / "topic_labels.json"
DATASET_CANDIDATES = [
    BASE_DIR / "data" / "course_evaluations_with_topics.csv",
    BASE_DIR / "data" / "processed_scaled_down_reviews_with_topics.csv",
    BASE_DIR / "data" / "processed_scaled_down_reviews.csv",
    BASE_DIR / "data" / "202511-ft_bi1_bi2_course_evaluation.csv",
    BASE_DIR / "course_evaluations.csv",
    BASE_DIR / "data.csv",
]
def ensure_nltk() -> None:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
def find_dataset_path() -> Path:
    for file_path in DATASET_CANDIDATES:
        if file_path.exists():
            return file_path
    raise FileNotFoundError(
        "Could not find a dataset. Update DATASET_CANDIDATES in app.py to match your project."
    )
def infer_text_column(df: pd.DataFrame) -> str:
    preferred = ["evaluation", "review", "comment", "feedback", "text"]
    cols_lower = {col.lower(): col for col in df.columns}
    for candidate in preferred:
        if candidate in cols_lower:
            return cols_lower[candidate]
    text_like_cols = [
        col
        for col in df.columns
        if col.lower().startswith("f_") and ("write" in col.lower() or "opinion" in col.lower())
    ]
    if text_like_cols:
        return text_like_cols[0]
    object_columns = [col for col in df.columns if df[col].dtype == "object"]
    if object_columns:
        return object_columns[0]
    raise ValueError("No suitable text column found in dataset.")
def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        lemmatizer.lemmatize(token)
        for token in text.split()
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(tokens)
def build_topic_labels(lda_model, vectorizer, top_n: int = 6) -> dict[str, str]:
    words = np.array(vectorizer.get_feature_names_out())
    labels: dict[str, str] = {}
    for topic_id, topic_weights in enumerate(lda_model.components_):
        top_words = words[np.argsort(topic_weights)[-top_n:]][::-1]
        labels[str(topic_id)] = f"Topic {topic_id}: " + ", ".join(top_words)
    return labels
def train_and_save_topic_artifacts(df_text: pd.Series, n_topics: int = 4):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned = df_text.fillna("").astype(str).apply(
        lambda value: preprocess_text(value, stop_words, lemmatizer)
    )
    cleaned = cleaned[cleaned.str.len() > 0]
    vectorizer = CountVectorizer(max_df=0.95, min_df=1)
    x_matrix = vectorizer.fit_transform(cleaned)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )
    lda_model.fit(x_matrix)
    topic_labels = build_topic_labels(lda_model, vectorizer, top_n=6)
    joblib.dump(lda_model, TOPIC_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with TOPIC_META_PATH.open("w", encoding="utf-8") as file:
        json.dump(topic_labels, file, indent=2)
    return lda_model, vectorizer, topic_labels
@st.cache_resource(show_spinner=False)
def load_resources():
    ensure_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    try:
        lda_model = joblib.load(TOPIC_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        if TOPIC_META_PATH.exists():
            with TOPIC_META_PATH.open("r", encoding="utf-8") as file:
                topic_labels = json.load(file)
        else:
            topic_labels = build_topic_labels(lda_model, vectorizer, top_n=6)
        return lda_model, vectorizer, topic_labels, stop_words, lemmatizer, sentiment_analyzer
    except Exception as exc:
        st.warning(
            f"Saved topic artifacts failed to load ({type(exc).__name__}). Retraining in current environment..."
        )
        dataset_path = find_dataset_path()
        df = pd.read_csv(dataset_path)
        text_col = infer_text_column(df)
        lda_model, vectorizer, topic_labels = train_and_save_topic_artifacts(df[text_col], n_topics=4)
        return lda_model, vectorizer, topic_labels, stop_words, lemmatizer, sentiment_analyzer
def predict_topic(
    text: str,
    lda_model,
    vectorizer,
    topic_labels,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
):
    cleaned = preprocess_text(text, stop_words, lemmatizer)
    if not cleaned:
        return "N/A (empty after preprocessing)", []
    x_matrix = vectorizer.transform([cleaned])
    probs = lda_model.transform(x_matrix)[0]
    topic_id = int(np.argmax(probs))
    label = topic_labels.get(str(topic_id), f"Topic {topic_id}")
    ranked = sorted(
        ((idx, prob) for idx, prob in enumerate(probs)),
        key=lambda item: item[1],
        reverse=True,
    )
    return label, ranked
def predict_sentiment(text: str, sentiment_analyzer: SentimentIntensityAnalyzer):
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "Positive", scores
    if compound <= -0.05:
        return "Negative", scores
    return "Neutral", scores
st.set_page_config(page_title="Course Evaluation Analyzer", page_icon="🧠", layout="centered")
st.title("Course Evaluation Analyzer")
st.write("Enter a student evaluation comment. The app predicts topic (LDA) and sentiment.")
lda_model, vectorizer, topic_labels, stop_words, lemmatizer, sentiment_analyzer = load_resources()
user_text = st.text_area(
    "Student evaluation text",
    height=150,
    placeholder="e.g., The lecturer explains concepts clearly but assignments are too heavy.",
)
if st.button("Analyze", type="primary"):
    if not user_text.strip():
        st.error("Please enter some text.")
    else:
        topic_label, topic_ranked = predict_topic(
            user_text,
            lda_model,
            vectorizer,
            topic_labels,
            stop_words,
            lemmatizer,
        )
        sentiment_label, sentiment_scores = predict_sentiment(user_text, sentiment_analyzer)
        st.subheader("Results")
        st.markdown(f"**Predicted Topic:** {topic_label}")
        st.markdown(f"**Predicted Sentiment:** {sentiment_label}")
        with st.expander("Details"):
            st.write("Topic probabilities:")
            for topic_id, probability in topic_ranked:
                st.write(f"- Topic {topic_id}: {probability:.3f}")
            st.write("Sentiment scores:", sentiment_scores)

import json
import re
from pathlib import Path
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
TOPIC_MODEL_PATH = MODEL_DIR / "topic_model_lda.pkl"
TOPIC_VECTORIZER_PATH = MODEL_DIR / "topic_vectorizer.pkl"
TOPIC_LABELS_PATH = MODEL_DIR / "topic_labels.json"
TOPIC_META_PATH = MODEL_DIR / "topic_artifact_metadata.json"
def ensure_nltk_data() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
def load_text_data() -> pd.Series:
    candidates = [
        DATA_DIR / "course_evaluations_with_topics.csv",
        DATA_DIR / "processed_scaled_down_reviews_with_topics.csv",
        DATA_DIR / "processed_scaled_down_reviews.csv",
        DATA_DIR / "202511-ft_bi1_bi2_course_evaluation.csv",
    ]
    for file_path in candidates:
        if not file_path.exists():
            continue
        data = pd.read_csv(file_path)
        if "text" in data.columns:
            return data["text"].dropna().astype(str)
        text_like_cols = [
            col
            for col in data.columns
            if col.lower().startswith("f_") and ("write" in col.lower() or "opinion" in col.lower())
        ]
        if text_like_cols:
            return data[text_like_cols].fillna("").astype(str).agg(" ".join, axis=1)
    raise FileNotFoundError("No dataset with usable text found in ./data")
def clean_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(cleaned_tokens)
def build_topic_labels(lda_model: LatentDirichletAllocation, vectorizer: CountVectorizer, top_n: int = 4) -> dict[int, str]:
    feature_names = vectorizer.get_feature_names_out()
    labels: dict[int, str] = {}
    for topic_id, topic_weights in enumerate(lda_model.components_):
        top_indices = topic_weights.argsort()[-top_n:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        labels[topic_id] = f"Topic {topic_id + 1}: {', '.join(top_words)}"
    return labels
def main() -> None:
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    texts = load_text_data().head(5000)
    cleaned_texts = [clean_text(t, stop_words, lemmatizer) for t in texts if str(t).strip()]
    if not cleaned_texts:
        raise ValueError("No non-empty text rows after preprocessing.")
    vectorizer = CountVectorizer(max_features=1500)
    x_matrix = vectorizer.fit_transform(cleaned_texts)
    lda_model = LatentDirichletAllocation(
        n_components=5,
        learning_method="batch",
        random_state=42,
        max_iter=12,
    )
    lda_model.fit(x_matrix)
    topic_labels = build_topic_labels(lda_model, vectorizer)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lda_model, TOPIC_MODEL_PATH)
    joblib.dump(vectorizer, TOPIC_VECTORIZER_PATH)
    with TOPIC_LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(topic_labels, f, indent=2)
    metadata = {
        "python": __import__("sys").version.split()[0],
        "numpy": __import__("numpy").__version__,
        "sklearn": __import__("sklearn").__version__,
        "n_components": 5,
        "max_features": 1500,
        "trained_rows": len(cleaned_texts),
    }
    with TOPIC_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {TOPIC_MODEL_PATH}")
    print(f"Saved {TOPIC_VECTORIZER_PATH}")
    print(f"Saved {TOPIC_LABELS_PATH}")
    print(f"Saved {TOPIC_META_PATH}")
    print("Sample labels:")
    for topic_id, label in topic_labels.items():
        print(f"  {topic_id}: {label}")
if __name__ == "__main__":
    main()

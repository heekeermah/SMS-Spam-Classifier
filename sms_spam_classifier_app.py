# app.py
import streamlit as st
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Set up NLTK cache directory (for deployment environments like Streamlit Cloud) ---
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
nltk.data.path.append(NLTK_DATA_DIR)

# --- Cache NLTK downloads to avoid repeated fetching ---
@st.cache_resource
def ensure_nltk_resources():
    # Clean corrupted or mislocated punkt_tab references (if any)
    from pathlib import Path
    punkt_tab_path = Path(NLTK_DATA_DIR) / "tokenizers" / "punkt_tab"
    if punkt_tab_path.exists():
        import shutil
        shutil.rmtree(punkt_tab_path)  # Remove broken directory

    # Now ensure correct downloads
    required = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet"
    }
    for resource, path in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, download_dir=NLTK_DATA_DIR)

ensure_nltk_resources()

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_classifier_and_vectorizer():
    try:
        model = joblib.load("spam_classifier.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load model or vectorizer: {e}")
        st.stop()

model, vectorizer = load_classifier_and_vectorizer()

# --- Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    words = word_tokenize(text)

    cleaned = [
        lemmatizer.lemmatize(word)
        for word in words if word.isalpha() and word not in stop_words
    ]

    return " ".join(cleaned)

# --- Streamlit UI ---
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.markdown("<h1 style='text-align: center;'>📩 SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("Enter an SMS message below to classify it as **Spam** or **Not Spam**.")

user_input = st.text_area("Enter the SMS message:", height=150,
                          placeholder="Type your message here...")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        try:
            cleaned_text = preprocess_text(user_input)
            input_vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(input_vector)

            if prediction[0] == 1:
                st.error("🚫 This message is **SPAM**!")
            else:
                st.success("✅ This message is **NOT SPAM**.")

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

st.markdown("---")
st.markdown("Created using Streamlit and Machine Learning (MultinomialNB + CountVectorizer).")

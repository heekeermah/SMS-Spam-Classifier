# app.py

import streamlit as st
import joblib
import os
import nltk
import shutil
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Set up NLTK data path (for compatibility with Streamlit Cloud) ---
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
nltk.data.path.append(NLTK_DATA_DIR)

# --- Clean and safely download required NLTK data ---
@st.cache_resource
def setup_nltk_resources():
    # Fix corrupted or unexpected "punkt_tab" path
    punkt_tab_path = Path(NLTK_DATA_DIR) / "tokenizers" / "punkt_tab"
    if punkt_tab_path.exists():
        shutil.rmtree(punkt_tab_path)

    # Ensure proper resources are downloaded
    required_resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet"
    }

    for resource, lookup_path in required_resources.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(resource, download_dir=NLTK_DATA_DIR)

setup_nltk_resources()

# --- Load trained model and vectorizer ---
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

# --- Text preprocessing function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Lowercase
    text = text.lower()

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(cleaned)

# --- Streamlit App UI ---
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
            # Preprocess input
            cleaned_text = preprocess_text(user_input)

            # Vectorize
            input_vector = vectorizer.transform([cleaned_text])

            # Predict
            prediction = model.predict(input_vector)

            if prediction[0] == 1:
                st.error("🚫 This message is **SPAM**!")
            else:
                st.success("✅ This message is **NOT SPAM**.")

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

st.markdown("---")
st.markdown("🔍 Built using Streamlit and Machine Learning (MultinomialNB + CountVectorizer).")

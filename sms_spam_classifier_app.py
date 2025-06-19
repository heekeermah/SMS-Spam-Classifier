# app.py

import streamlit as st
import joblib
import os
import re

# --- Minimal stopwords list to avoid nltk dependency ---
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
    'them', 'what', 'which', 'who', 'whom', 'this', 'that', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'just', 'should', 'now', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until'
])

# --- Load trained model and vectorizer ---
@st.cache_resource
def load_classifier_and_vectorizer():
    try:
        model = joblib.load("spam_classifier (2).pkl")
        vectorizer = joblib.load("vectorizer (1).pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load model or vectorizer: {e}")
        st.stop()

model, vectorizer = load_classifier_and_vectorizer()

# --- Text preprocessing function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    words = re.findall(r'\b[a-z]{2,}\b', text)
    return " ".join([word for word in words if word not in stop_words])

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
st.markdown("🔍 Built using Streamlit and Machine Learning (MultinomialNB + CountVectorizer).")

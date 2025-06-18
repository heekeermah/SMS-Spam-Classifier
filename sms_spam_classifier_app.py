# app.py
import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Configuration ---
MODEL_PATH = "spam_classifier.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
NLTK_DATA_DIR = "/tmp/nltk_data" # A common writable directory for NLTK data in cloud environments

# Ensure the NLTK data directory exists
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Set NLTK_DATA environment variable and add to NLTK's search path EARLY
# This is crucial to ensure NLTK can find the downloaded data from the start
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
nltk.data.path.append(NLTK_DATA_DIR)

# --- Download NLTK data (Crucial for deployment) ---
@st.cache_resource(show_spinner=False) # Hide spinner for NLTK download
def download_nltk_data_safe():
    """
    Attempts to download necessary NLTK data to a specified directory.
    Returns True on success, False on failure.
    """
    try:
        # Download to the specified directory using download_dir argument
        # Removed quiet=True for better error visibility in Streamlit logs
        print(f"Attempting to download NLTK data to: {NLTK_DATA_DIR}")
        nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
        nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
        nltk.download('punkt', download_dir=NLTK_DATA_DIR)
        print("NLTK data download complete.")
        return True
    except Exception as e:
        # Print the exception to the console for debugging in Streamlit logs
        print(f"Error during NLTK download: {e}")
        return False

# Attempt to download NLTK data at startup
nltk_data_ready = download_nltk_data_safe()

# If NLTK data failed to download, display an error and stop the app
if not nltk_data_ready:
    st.error("Failed to download necessary NLTK data. "
             "This often indicates a network issue or an inability to write to the designated directory. "
             "Please check your Streamlit app logs for more details.")
    st.stop()

# Initialize NLTK components for preprocessing
ps = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
# This should now load correctly as data path is set and download attempted
def preprocess_text(text):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
        stop_words = set(stopwords.words('english'))

# --- Load Model and Vectorizer ---
@st.cache_resource # Cache the loading of the model and vectorizer
def load_classifier_and_vectorizer():
    """
    Loads the trained spam classifier (MultinomialNB) and CountVectorizer.
    Uses st.cache_resource to avoid reloading on every rerun.
    """
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Error: Could not find '{MODEL_PATH}' or '{VECTORIZER_PATH}'. "
                 "Please ensure these files are in the same directory as app.py.")
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred while loading the model or vectorizer: {e}")
        st.stop()

model, vectorizer = load_classifier_and_vectorizer()

# --- Text Preprocessing Function (Matches Notebook exactly) ---
def preprocess_text(text):
    """
    Applies the full preprocessing pipeline as described in the notebook:
    1. Lowercasing
    2. Tokenization
    3. Removing non-alphabetic characters
    4. Stop word removal
    5. Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercasing
    text = text.lower()

    # 2. Tokenization
    # This will now correctly look for punkt data in NLTK_DATA_DIR due to nltk.data.path.append
    words = word_tokenize(text)

    # 3. Removing non-alphabetic characters and 4. Stop word removal
    # The notebook's approach combines this with lemmatization for efficiency
    processed_words = []
    for word in words:
        if word.isalpha() and word not in stop_words: # Keep only alphabetic and non-stopwords
            # 5. Lemmatization
            processed_words.append(wordnet_lemmatizer.lemmatize(word))

    return " ".join(processed_words)

# --- Streamlit UI ---
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stTextArea label {
        font-size: 1.1em;
        font-weight: bold;
        color: #333;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 10px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 10px;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📩 SMS Spam Classifier")
st.markdown("Enter an SMS message below to classify it as 'Spam' or 'Not Spam'.")

# User input
user_input = st.text_area("Enter the SMS message:", height=150,
                          placeholder="Type your message here...")

# Predict button
if st.button("Classify"):
    if not user_input or user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess the user input using the exact notebook logic
        processed_input = preprocess_text(user_input)

        # 2. Transform the processed input using the loaded CountVectorizer
        try:
            # The vectorizer expects an iterable (e.g., a list) of texts.
            input_vector = vectorizer.transform([processed_input])
        except Exception as e:
            st.error(f"Error during text transformation: {e}. "
                     "Ensure your vectorizer is loaded correctly and matches training data expectations.")
            st.stop()

        # 3. Make prediction
        try:
            prediction = model.predict(input_vector)

            # Based on your notebook, 'ham' is 0 and 'spam' is 1
            predicted_label = "spam" if prediction[0] == 1 else "NOT SPAM"

        except Exception as e:
            st.error(f"Error during prediction: {e}. "
                     "Ensure your model is loaded correctly and matches vectorizer output expectations.")
            st.stop()

        # 4. Display result
        if predicted_label == "spam":
            st.error("🚫 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")

st.markdown("---")
st.markdown("Developed with Streamlit and Machine Learning for SMS classification.")

"""Streamlit web application for SpamGuard."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "spam_classifier.pkl"
VECTORIZER_FILE = BASE_DIR / "vectorizer.pkl"

st.set_page_config(
    page_title="SpamGuard | AI Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
        return None, None
    return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_message(message: str, classifier, vectorizer):
    cleaned = clean_text(message)
    transformed = vectorizer.transform([cleaned])
    prediction = classifier.predict(transformed)[0]
    probabilities = classifier.predict_proba(transformed)[0]
    classes = list(classifier.classes_)
    probability_map = dict(zip(classes, probabilities))
    confidence = probability_map[prediction]
    return prediction, confidence, probability_map


# --------------------------- Custom styling ---------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        .hero {
            padding: 1.8rem 2rem;
            border-radius: 22px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.90));
            color: white;
            margin-bottom: 1.3rem;
        }
        .hero h1 {margin: 0; font-size: 2.6rem; letter-spacing: -0.04em;}
        .hero p {margin: 0.45rem 0 0; color: #cbd5e1; font-size: 1.04rem;}
        .result-card {
            padding: 1.2rem 1.35rem;
            border-radius: 18px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(127, 127, 127, 0.07);
            margin-top: 1rem;
        }
        .result-label {font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7;}
        .result-value {font-size: 2rem; font-weight: 750; margin-top: 0.15rem;}
        .muted {opacity: 0.68; font-size: 0.9rem;}
        .footer {text-align: center; opacity: 0.55; margin-top: 2.5rem; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


classifier, vectorizer = load_artifacts()

with st.sidebar:
    st.markdown("## 🛡️ SpamGuard")
    st.caption("NLP-based spam text classifier")
    st.divider()
    st.markdown("### Model")
    st.write("**TF-IDF** feature extraction")
    st.write("**Logistic Regression** classifier")
    st.write("**Random state:** 42")
    st.divider()
    st.markdown("### Dataset")
    st.write("UCI SMS Spam Collection")
    st.write("5,574 labeled messages")
    st.caption("The training corpus contains SMS messages; the model is a text-spam classifier and may behave differently on formal email datasets.")

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SpamGuard</h1>
        <p>AI-powered spam detection using Natural Language Processing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if classifier is None or vectorizer is None:
    st.error("Model files are missing. Run `py -3.12 train_model.py` first, then start the app again.")
    st.stop()

left, right = st.columns([1.35, 0.65], gap="large")

with left:
    st.subheader("Analyze a message")
    st.write("Paste an email, SMS, or other text below and SpamGuard will classify it.")

    message = st.text_area(
        "Message",
        height=210,
        placeholder="Example: Congratulations! You have won a free prize. Click the link to claim it...",
        label_visibility="collapsed",
    )

    sample_left, sample_right = st.columns(2)
    with sample_left:
        if st.button("Try a spam example", use_container_width=True):
            message = "Congratulations! You have won a cash prize. Click this link now to claim your reward."
    with sample_right:
        if st.button("Try a normal example", use_container_width=True):
            message = "Hi, are we still meeting at the library after class?"

    analyze = st.button("🔎 Analyze Message", type="primary", use_container_width=True)

    if analyze:
        if not message.strip():
            st.warning("Please enter a message before analyzing it.")
        else:
            prediction, confidence, probability_map = classify_message(message, classifier, vectorizer)
            is_spam = prediction == "spam"
            result_text = "SPAM" if is_spam else "NOT SPAM"

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Prediction</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">{result_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">Model confidence: {confidence * 100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.progress(float(confidence), text=f"Confidence: {confidence * 100:.1f}%")

            ham_probability = probability_map.get("ham", 0.0)
            spam_probability = probability_map.get("spam", 0.0)

            metric1, metric2 = st.columns(2)
            metric1.metric("Not Spam", f"{ham_probability * 100:.1f}%")
            metric2.metric("Spam", f"{spam_probability * 100:.1f}%")

            if is_spam:
                st.warning("This message contains patterns the model associates with spam. Treat links, prizes, urgent requests, and unknown senders cautiously.")
            else:
                st.success("The model does not detect strong spam patterns in this message.")

            st.caption("Confidence is the model's probability estimate, not a guarantee. Real-world spam can be difficult to classify.")

with right:
    st.subheader("How it works")
    st.markdown(
        """
        **1. Text cleaning**  
        Normalizes URLs, email addresses, numbers, punctuation, and whitespace.

        **2. TF-IDF**  
        Converts message text into numerical features based on word and phrase importance.

        **3. Classification**  
        Logistic Regression predicts `spam` or `ham` from the TF-IDF features.

        **4. Confidence**  
        The application displays the classifier's probability estimate for the selected class.
        """
    )

    st.info("Tip: test the model with several different examples rather than relying on a single prediction.")

st.markdown(
    '<div class="footer">SpamGuard • NLP + Machine Learning • Built with Python, scikit-learn and Streamlit</div>',
    unsafe_allow_html=True,
)

import streamlit as st
import joblib
from pathlib import Path

# -----------------------------
# Load Model and TF-IDF
# -----------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# -----------------------------
# Header
# -----------------------------
st.title("📰 Fake News Detection System")

st.write(
    "An NLP-based machine learning system that classifies "
    "news articles as **Fake News** or **Real News**."
)

st.divider()


# -----------------------------
# Model Information
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.info("🤖 **Algorithm**\n\nLogistic Regression")

with col2:
    st.info("🔤 **Feature Extraction**\n\nTF-IDF")


st.divider()


# -----------------------------
# News Input
# -----------------------------
st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste the news article below:",
    height=250,
    placeholder="Example: The government announced a new policy today..."
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article first.")

    else:

        # Convert text into TF-IDF
        news_tfidf = tfidf.transform([news_text])

        # Make prediction
        prediction = lr_model.predict(news_tfidf)[0]

        # Get probabilities
        probabilities = lr_model.predict_proba(news_tfidf)[0]

        fake_probability = probabilities[0] * 100
        real_probability = probabilities[1] * 100

        st.divider()

        # -----------------------------
        # Prediction Result
        # -----------------------------
        st.subheader("🎯 Prediction Result")

        if prediction == 0:
            st.error("🚨 **Prediction: FAKE NEWS**")
        else:
            st.success("✅ **Prediction: REAL NEWS**") 

        st.warning(
    "⚠️ This is a machine-learning prediction, not a factual verification. "
    "The system does not verify the news against the Internet or external sources."
)


        # -----------------------------
        # Confidence
        # -----------------------------
        st.subheader("📊 Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🚨 Fake News",
                f"{fake_probability:.2f}%"
            )
            st.progress(fake_probability / 100)

        with col2:
            st.metric(
                "✅ Real News",
                f"{real_probability:.2f}%"
            )
            st.progress(real_probability / 100)


        # -----------------------------
        # Model Accuracy
        # -----------------------------
        # -----------------------------
# Model Performance
# -----------------------------
st.divider()

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "96.02%")

with col2:
    st.metric("Precision", "95.95%")

with col3:
    st.metric("Recall", "95.23%")

with col4:
    st.metric("F1 Score", "95.59%")
        


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "⚠️ This system provides predictions based on patterns learned "
    "from the training dataset. It should not be considered a replacement "
    "for professional fact-checking."
)

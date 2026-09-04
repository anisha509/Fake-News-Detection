import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# LOAD MODEL AND TF-IDF

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    lr_model = joblib.load(BASE_DIR / "logistic_regression_model.pkl")
    return tfidf, lr_model


tfidf, lr_model = load_models()


# PROJECT INFORMATION

PROJECT_TITLE = "📰 Fake News Detection System"

DATASET_ROWS = 72134
DATASET_COLUMNS = 4

TRAINING_SAMPLES = 50897
TESTING_SAMPLES = 12725

TFIDF_FEATURES = 50000

DUPLICATE_CONTENT = 8456
CONFLICTING_DUPLICATE_LABELS = 0
EMPTY_CONTENT = 0

TRAIN_FAKE = 27832
TRAIN_REAL = 23065

TEST_FAKE = 6959
TEST_REAL = 5766

ACCURACY = 0.9602
PRECISION = 0.9595
RECALL = 0.9523
F1_SCORE = 0.9559

CONFUSION_MATRIX = [
    [6727, 232],
    [275, 5491]
]


# SIDEBAR

st.sidebar.title("📰 Fake News Detection")

st.sidebar.markdown("""
### Navigation

Use the sections below to explore the project.
""")

section = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📊 Dataset & Analysis",
        "🔍 Prediction",
        "📈 Model Performance",
        "💡 Conclusion"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Project\n\n"
    "TF-IDF + Logistic Regression"
)

# HEADER

st.title(PROJECT_TITLE)

st.markdown(
    """
    ### NLP-Based Fake News Classification using Logistic Regression

    This project uses **Natural Language Processing (NLP)** and
    **Machine Learning** to classify news articles as **Fake News**
    or **Real News**.

    **Technique:** TF-IDF Vectorization  
    **Algorithm:** Logistic Regression
    """
)

st.markdown("---")


# OVERVIEW

if section == "🏠 Overview":

    st.header("📌 Project Overview")

    st.write(
        """
        Fake news can spread rapidly through online platforms and can
        influence public opinion. The objective of this project is to
        develop a machine learning system that classifies news content
        into two categories:

        - 🚨 **Fake News**
        - ✅ **Real News**

        The text is converted into numerical features using **TF-IDF
        (Term Frequency-Inverse Document Frequency)** and then classified
        using **Logistic Regression**.
        """
    )

    st.subheader("🎯 Project Objective")

    st.write(
        """
        To develop an NLP-based machine learning model capable of
        automatically classifying news articles as fake or real and
        provide an interactive dashboard for analyzing the model results.
        """
    )

    st.subheader("⚙️ Methodology")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("1️⃣ Preprocessing", "Text Cleaning")

    with col2:
        st.metric("2️⃣ Feature Extraction", "TF-IDF")

    with col3:
        st.metric("3️⃣ ML Algorithm", "Logistic Regression")

    with col4:
        st.metric("4️⃣ Evaluation", "Classification Metrics")

    st.subheader("🔄 Project Workflow")

    st.info(
        """
        News Text  
        ↓  
        Text Preprocessing  
        ↓  
        TF-IDF Vectorization  
        ↓  
        Logistic Regression  
        ↓  
        Fake / Real Prediction  
        ↓  
        Performance Analysis
        """
    )

    st.subheader("⚠️ Important Disclaimer")

    st.warning(
        "This system is a machine-learning text classifier. "
        "It does not verify news against the Internet, government sources, "
        "or external fact-checking websites."
    )


# ============================================================
# DATASET & ANALYSIS
# ============================================================

elif section == "📊 Dataset & Analysis":

    st.header("📊 Dataset Overview")

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Dataset Records",
            f"{DATASET_ROWS:,}"
        )

    with col2:
        st.metric(
            "Training Samples",
            f"{TRAINING_SAMPLES:,}"
        )

    with col3:
        st.metric(
            "Testing Samples",
            f"{TESTING_SAMPLES:,}"
        )

    with col4:
        st.metric(
            "TF-IDF Features",
            f"{TFIDF_FEATURES:,}"
        )

    st.markdown("---")
    
    # Dataset columns

    st.subheader("📋 Dataset Features")

    dataset_features = pd.DataFrame({
        "Feature": [
            "title",
            "text",
            "label"
        ],
        "Description": [
            "Title of the news article",
            "Main text/content of the news article",
            "Target class: 0 = Fake, 1 = Real"
        ]
    })

    st.dataframe(
        dataset_features,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The original dataset contained an additional index column "
        "(`Unnamed: 0`), which was removed during preprocessing."
    )

    # Key statistics

    st.subheader("📌 Key Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Duplicate Content",
            f"{DUPLICATE_CONTENT:,}"
        )

    with col2:
        st.metric(
            "Conflicting Duplicates",
            f"{CONFLICTING_DUPLICATE_LABELS}"
        )

    with col3:
        st.metric(
            "Empty Content",
            f"{EMPTY_CONTENT}"
        )

    with col4:
        st.metric(
            "Model",
            "Logistic Regression"
        )

    st.markdown("---")

    # Class Distribution

    st.subheader("📊 Sentiment / Class Distribution")

    st.write(
        "This project uses **class distribution** rather than sentiment "
        "analysis because the target classes are Fake News and Real News."
    )

    split = st.selectbox(
        "Select dataset split",
        ["Training Set", "Testing Set"]
    )

    if split == "Training Set":

        class_data = pd.DataFrame({
            "Class": ["Fake News", "Real News"],
            "Count": [TRAIN_FAKE, TRAIN_REAL]
        })

    else:

        class_data = pd.DataFrame({
            "Class": ["Fake News", "Real News"],
            "Count": [TEST_FAKE, TEST_REAL]
        })

    fig_class = px.bar(
        class_data,
        x="Class",
        y="Count",
        title=f"{split} Class Distribution",
        text="Count"
    )

    fig_class.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_class,
        use_container_width=True
    )

    # Training vs Testing Samples

    st.subheader("📈 Training vs Testing Data")

    split_data = pd.DataFrame({
        "Dataset": [
            "Training",
            "Testing"
        ],
        "Samples": [
            TRAINING_SAMPLES,
            TESTING_SAMPLES
        ]
    })

    fig_split = px.pie(
        split_data,
        names="Dataset",
        values="Samples",
        title="Training and Testing Dataset Distribution"
    )

    st.plotly_chart(
        fig_split,
        use_container_width=True
    )
    
    # Word Clouds

    st.subheader("☁️ Word Cloud")

    st.write(
        """
        The word clouds below visualize the most influential terms
        learned by the Logistic Regression model.

        - Positive coefficients → terms associated with **Real News**
        - Negative coefficients → terms associated with **Fake News**
        """
    )

    feature_names = tfidf.get_feature_names_out()
    coefficients = lr_model.coef_[0]

    # Top Real-associated terms
    real_indices = coefficients.argsort()[-100:][::-1]

    real_frequencies = {
        feature_names[i]: float(coefficients[i])
        for i in real_indices
        if coefficients[i] > 0
    }

    # Top Fake-associated terms
    fake_indices = coefficients.argsort()[:100]

    fake_frequencies = {
        feature_names[i]: float(abs(coefficients[i]))
        for i in fake_indices
        if coefficients[i] < 0
    }

    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:

        st.markdown("### ✅ Real News Associated Terms")

        if real_frequencies:

            real_wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate_from_frequencies(real_frequencies)

            fig_real, ax_real = plt.subplots(
                figsize=(10, 5)
            )

            ax_real.imshow(
                real_wordcloud,
                interpolation="bilinear"
            )

            ax_real.axis("off")

            st.pyplot(
                fig_real,
                clear_figure=True
            )

    with wc_col2:

        st.markdown("### 🚨 Fake News Associated Terms")

        if fake_frequencies:

            fake_wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate_from_frequencies(fake_frequencies)

            fig_fake, ax_fake = plt.subplots(
                figsize=(10, 5)
            )

            ax_fake.imshow(
                fake_wordcloud,
                interpolation="bilinear"
            )

            ax_fake.axis("off")

            st.pyplot(
                fig_fake,
                clear_figure=True
            )


# PREDICTION

elif section == "🔍 Prediction":

    st.header("🔍 Fake News Prediction")

    st.write(
        """
        Enter a news article below and the trained Logistic Regression
        model will classify it as Fake News or Real News.
        """
    )

    news_text = st.text_area(
        "📝 Enter News Article",
        height=250,
        placeholder="Paste the news article here..."
    )

    if st.button(
        "🔍 Check News",
        use_container_width=True
    ):

        if not news_text.strip():

            st.warning(
                "⚠️ Please enter some news text first."
            )

        else:

            # Transform text
            news_tfidf = tfidf.transform(
                [news_text]
            )

            # Prediction
            prediction = int(
                lr_model.predict(news_tfidf)[0]
            )

            # Probabilities
            probabilities = lr_model.predict_proba(
                news_tfidf
            )[0]

            class_probabilities = dict(
                zip(
                    lr_model.classes_,
                    probabilities
                )
            )

            fake_probability = (
                class_probabilities.get(0, 0) * 100
            )

            real_probability = (
                class_probabilities.get(1, 0) * 100
            )

            st.markdown("---")

            # Prediction result
            if prediction == 0:

                st.error(
                    "🚨 Prediction: FAKE NEWS"
                )

            else:

                st.success(
                    "✅ Prediction: REAL NEWS"
                )

            # Probability metrics
            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🚨 Fake News Probability",
                    f"{fake_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "✅ Real News Probability",
                    f"{real_probability:.2f}%"
                )

            # Probability chart
            probability_data = pd.DataFrame({
                "Class": [
                    "Fake News",
                    "Real News"
                ],
                "Probability": [
                    fake_probability,
                    real_probability
                ]
            })

            fig_probability = px.bar(
                probability_data,
                x="Class",
                y="Probability",
                title="Prediction Probability",
                text="Probability"
            )

            fig_probability.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig_probability.update_yaxes(
                range=[0, 100]
            )

            st.plotly_chart(
                fig_probability,
                use_container_width=True
            )

            st.warning(
                "⚠️ This is a machine-learning prediction, not a factual "
                "verification. The system does not verify the news against "
                "the Internet or external sources."
            )

# MODEL PERFORMANCE

elif section == "📈 Model Performance":

    st.header("📈 Model Performance")

    st.write(
        """
        The Logistic Regression model was evaluated using Accuracy,
        Precision, Recall and F1 Score.
        """
    )

    # Metrics
   
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "96.02%"
        )

    with col2:
        st.metric(
            "Precision",
            "95.95%"
        )

    with col3:
        st.metric(
            "Recall",
            "95.23%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "95.59%"
        )

    # Performance comparison
   
    st.subheader("📊 Model Performance Comparison")

    performance_data = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            ACCURACY * 100,
            PRECISION * 100,
            RECALL * 100,
            F1_SCORE * 100
        ]
    })

    fig_performance = px.bar(
        performance_data,
        x="Metric",
        y="Score",
        title="Logistic Regression Performance",
        text="Score"
    )

    fig_performance.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_performance.update_yaxes(
        range=[0, 100],
        title="Score (%)"
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True
    )

    # Confusion Matrix
 
    st.subheader("🔢 Confusion Matrix")

    st.write(
        """
        The confusion matrix shows the number of correct and incorrect
        predictions made by the model.
        """
    )

    cm = CONFUSION_MATRIX

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[
                "Predicted Fake",
                "Predicted Real"
            ],
            y=[
                "Actual Fake",
                "Actual Real"
            ],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Count: %{z}<extra></extra>"
            )
        )
    )

    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    # Feature Importance
  
    st.subheader("🔎 Feature Importance Analysis")

    feature_names = tfidf.get_feature_names_out()
    coefficients = lr_model.coef_[0]

    # Top 10 positive coefficients = Real
    top_real_indices = coefficients.argsort()[-10:][::-1]

    real_features = pd.DataFrame({
        "Feature": [
            feature_names[i]
            for i in top_real_indices
        ],
        "Coefficient": [
            coefficients[i]
            for i in top_real_indices
        ],
        "Class": "Real News"
    })

    # Top 10 negative coefficients = Fake
    top_fake_indices = coefficients.argsort()[:10]

    fake_features = pd.DataFrame({
        "Feature": [
            feature_names[i]
            for i in top_fake_indices
        ],
        "Coefficient": [
            coefficients[i]
            for i in top_fake_indices
        ],
        "Class": "Fake News"
    })

    feature_data = pd.concat(
        [
            real_features,
            fake_features
        ],
        ignore_index=True
    )

    fig_features = px.bar(
        feature_data,
        x="Coefficient",
        y="Feature",
        color="Class",
        orientation="h",
        title="Top Important Features"
    )

    st.plotly_chart(
        fig_features,
        use_container_width=True
    )

    st.info(
        """
        Interpretation:

        Positive coefficients are associated with **Real News (label 1)**,
        while negative coefficients are associated with
        **Fake News (label 0)**.
        """
    )


# CONCLUSION

elif section == "💡 Conclusion":

    st.header("💡 Conclusion")

    # Key Findings
 

    st.subheader("🔑 Key Findings")

    st.write(
        """
        - The Fake News Detection system achieved approximately
          **96.02% accuracy**.
        - Precision was approximately **95.95%**.
        - Recall was approximately **95.23%**.
        - The F1 Score was approximately **95.59%**.
        - TF-IDF successfully converted news text into numerical
          features suitable for machine learning.
        - Logistic Regression provided strong classification performance
          for the fake news detection task.
        """
    )
    # Challenges

    st.subheader("⚠️ Challenges Faced")

    st.write(
        """
        1. News articles contain noisy and unstructured text.
        2. Text preprocessing is required before machine learning.
        3. TF-IDF produces a high-dimensional feature space.
        4. Duplicate content needed to be identified.
        5. The model cannot independently verify whether a news claim
           is factually true.
        """
    )


    # Future Scope

    st.subheader("🚀 Future Scope")

    st.write(
        """
        - Use advanced NLP models such as BERT or transformer-based models.
        - Add real-time fact-checking using trusted external sources.
        - Support multiple languages.
        - Integrate social media and online news monitoring.
        - Improve detection of newly emerging misinformation patterns.
        - Deploy the system as a larger real-time web application.
        """
    )

    # Applications

    st.subheader("💼 Applications of the Project")

    st.write(
        """
        - 📰 News verification assistance
        - 📱 Social media content monitoring
        - 🎓 Educational awareness systems
        - 🛡️ Misinformation screening
        - 💻 Content moderation support
        - 🔎 Journalism and research assistance
        """
    )

    st.markdown("---")

    st.success(
        """
        ### 🎯 Final Conclusion

        The project demonstrates that Natural Language Processing combined
        with TF-IDF and Logistic Regression can effectively classify news
        articles into Fake and Real categories. The interactive Streamlit
        dashboard provides a convenient way to explore the dataset,
        visualize model performance and test new news articles.
        """
    )

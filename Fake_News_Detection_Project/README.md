📰 Fake News Detection using Logistic Regression

📌 Project Overview

This project is an NLP-based Fake News Detection System that classifies news articles as **Fake News** or **Real News** using Machine Learning.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text feature extraction and **Logistic Regression** as the classification algorithm.

A **Streamlit web application** is also developed to allow users to enter news text and receive a model prediction with confidence scores.

 ⚠️ The system is a machine-learning text classifier. It does not verify news against the Internet or external fact-checking sources.

 🎯 Objective

The main objectives of this project are:

- To detect whether a news article is likely to be Fake or Real.
- To apply Natural Language Processing techniques to news text.
- To convert textual data into numerical features using TF-IDF.
- To train a Logistic Regression classification model.
- To analyze important features learned by the model.
- To develop an interactive Streamlit application for predictions.


 📊 Dataset

The project uses the *WELFake Dataset*.

The original dataset contains:

- 72,134 records
- News title
- News text
- Label

 Label Mapping

| Label | Class |
|------:|-------|
| 0 | Fake News |
| 1 | Real News |

---

 🔄 Project Workflow

```text
WELFake Dataset
       ↓
Data Cleaning
       ↓
Combine Title + Text
       ↓
Train/Test Split
       ↓
TF-IDF Feature Extraction
       ↓
Logistic Regression
       ↓
Model Evaluation
       ↓
Feature Importance Analysis
       ↓
Streamlit Web Application

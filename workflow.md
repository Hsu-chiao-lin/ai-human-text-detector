# HW5 Workflow — AI vs Human Text Detector

## 1. Problem Definition
The goal of this project is to build a simple system that can distinguish between AI-generated and human-written text.

## 2. Data Collection
- AI texts were generated using ChatGPT.
- Human texts were collected from articles and personal writing samples.
- Each text was labeled as AI (1) or Human (0).

## 3. Feature Engineering
Handcrafted linguistic features were extracted, including:
- Average sentence length
- Total word count
- Vocabulary diversity
- Punctuation ratio
- Average word length

## 4. Model Training
A Logistic Regression classifier from scikit-learn was used.
The dataset was split into training and testing sets to evaluate accuracy.

## 5. Deployment
The trained model was integrated into a Streamlit web application.
Users can input text and receive AI/Human probability scores in real time.

## 6. Result
The system successfully provides an interpretable AI detection result via a deployed Streamlit app.


# AI vs Human Text Detector

## Introduction
This project implements a simple AI-generated text detector using handcrafted linguistic features and a machine learning classifier.

## Method
- Feature engineering (sentence length, vocabulary diversity, punctuation ratio)
- Logistic Regression classifier (scikit-learn)

## Demo
Streamlit App: https://aihumandetector.streamlit.app/

## How to Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py

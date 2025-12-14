import streamlit as st
import joblib
from features import extract_features

st.set_page_config(page_title="AI Detector", layout="centered")

st.title("📝 AI vs Human Text Detector")

st.write("Paste your text below to analyze whether it is AI-generated or human-written.")

model = joblib.load("model.pkl")

text = st.text_area("Input Text", height=200)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        features = extract_features(text)
        prob = model.predict_proba([features])[0]

        st.subheader("Detection Result")
        st.metric("AI-generated Probability", f"{prob[1]*100:.2f}%")
        st.metric("Human-written Probability", f"{prob[0]*100:.2f}%")

        st.progress(int(prob[1] * 100))

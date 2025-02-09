import streamlit as st
import pickle
import pandas as pd

# Load the trained model and vectorizer
with open("D:\\MIT study\\NLPT\\New folder\\sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("D:\\MIT study\\NLPT\\New folder\\tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("Sentiment Analysis App")
st.write("Enter a sentence and the model will predict if it's positive or negative.")

user_input = st.text_area("Enter text here:")

if st.button("Analyze Sentiment"):
    if user_input:
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
        st.subheader(f"Prediction: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")


# Footer
st.markdown("🚀 Developed by Surabhi")

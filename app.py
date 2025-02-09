import streamlit as st
import pickle
import os
import pandas as pd

# Function to load the model and vectorizer
def load_pickle(file_name):
    try:
        with open(file_name, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: {file_name} not found. Ensure it's uploaded to your repository.")
        return None

# Load model and vectorizer from the current directory
model_path = os.path.join(os.getcwd(), "sentiment_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "tfidf_vectorizer.pkl")

model = load_pickle(model_path)
vectorizer = load_pickle(vectorizer_path)

# Streamlit UI
st.title("📊 Sentiment Analysis App")
st.write("Enter a sentence and the model will predict if it's **positive or negative**.")

user_input = st.text_area("Enter text here:")

if st.button("Analyze Sentiment"):
    if model and vectorizer:  # Ensure model & vectorizer are loaded
        if user_input:
            text_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(text_vectorized)[0]
            sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
            st.success(f"**Prediction:** {sentiment}")
        else:
            st.warning("⚠️ Please enter some text for analysis.")
    else:
        st.error("🚨 Model and vectorizer could not be loaded. Please check deployment.")

# Footer
st.markdown("🚀 Developed by **Surabhi**")

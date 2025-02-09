import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("D:\\MIT study\\NLPT\\New folder\\sentiment_model.pkl")
vectorizer = joblib.load("D:\\MIT study\\NLPT\\New folder\\tfidf_vectorizer.pkl")

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and check its sentiment!")

# User input
user_input = st.text_area("Enter your review here...", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Transform input using the saved vectorizer
        input_vectorized = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(input_vectorized)[0]
        sentiment = "😃 Positive" if prediction == 1 else "😞 Negative"

        # Display result
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review before analyzing.")

# Footer
st.markdown("🚀 Developed by Surabhi")

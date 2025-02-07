import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Streamlit app title
st.title("Review Sentiment Analysis")

# Input text box
text_input = st.text_input("Write your comment")

# Button to trigger sentiment analysis
if st.button("Get sentiment"):
    if text_input.strip():  # Ensure input is not empty
        # Tokenize the input text
        tokens = tokenizer.encode(text_input, return_tensors='pt')

        # Get model prediction
        result = model(tokens)

        # Convert logits to a score (1-5 scale)
        score = int(torch.argmax(result.logits)) + 1
        # Map score to sentiment label
        if score in [1, 2]:
            sentiment = "Negative"
        elif score == 3:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"

        # Display results
        st.write(f"Score on a scale of 1-5: {score}")
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a comment to analyze.")

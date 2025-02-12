import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-snli')

# Function to preprocess and get sentence embeddings
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output[0][:, 0, :].squeeze().numpy()  # Extract embeddings from the [CLS] token
    return embeddings

# Function to compute similarity and predict NLI
def predict_nli(premise, hypothesis):
    premise_embedding = get_sentence_embedding(premise)
    hypothesis_embedding = get_sentence_embedding(hypothesis)
    
    # Cosine similarity between the sentence embeddings
    similarity = cosine_similarity([premise_embedding], [hypothesis_embedding])[0][0]
    
    # Predicted NLI labels based on cosine similarity
    if similarity > 0.7:
        return "Entailment"
    elif similarity > 0.3:
        return "Neutral"
    else:
        return "Contradiction"

# Streamlit UI for input and displaying results
st.title("Text Similarity and NLI Prediction")
st.write("Enter two sentences below to predict their relationship (Entailment, Neutral, or Contradiction).")

premise = st.text_input("Premise Sentence", "")
hypothesis = st.text_input("Hypothesis Sentence", "")

if st.button("Predict"):
    if premise and hypothesis:
        result = predict_nli(premise, hypothesis)
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter both sentences for prediction.")


import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load your custom-trained sentence-transformer model
# For example, we're using a pre-trained model here, but you can replace it with your custom model path
model = SentenceTransformer('all-MiniLM-L6-v2')  # replace this with your custom-trained model path

# Function to get sentence embeddings
def get_sentence_embedding(sentence):
    # Get the embedding of the sentence
    embedding = model.encode([sentence])[0]
    return embedding

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

# Input boxes for the premise and hypothesis
premise = st.text_input("Premise Sentence", "")
hypothesis = st.text_input("Hypothesis Sentence", "")

# Predict button
if st.button("Predict"):
    if premise and hypothesis:
        result = predict_nli(premise, hypothesis)
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter both sentences for prediction.")

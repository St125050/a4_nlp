import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from bert_model import mean_pool

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and tokenizer from Hugging Face
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def classify_nli(model, tokenizer, sentence_a, sentence_b, device):
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    labels = ['Entailment', 'Neutral', 'Contradiction']
    result = {label: prob.item() for label, prob in zip(labels, probabilities)}
    return result

# Streamlit App
st.title("NLI with BERT")

sentence_a = st.text_input("Enter the first sentence:")
sentence_b = st.text_input("Enter the second sentence:")

if st.button("Classify NLI"):
    if sentence_a and sentence_b:
        result = classify_nli(model, tokenizer, sentence_a, sentence_b, device)
        st.write(f"Entailment: {result['Entailment']:.4f}")
        st.write(f"Neutral: {result['Neutral']:.4f}")
        st.write(f"Contradiction: {result['Contradiction']:.4f}")
    else:
        st.write("Please enter both sentences to classify NLI.")

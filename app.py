import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from bert_model import mean_pool

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and tokenizer from Hugging Face
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    inputs_a = tokenizer(sentence_a, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    inputs_b = tokenizer(sentence_b, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    inputs_ids_a = inputs_a['input_ids']
    attention_a = inputs_a['attention_mask']
    inputs_ids_b = inputs_b['input_ids']
    attention_b = inputs_b['attention_mask']

    with torch.no_grad():
        u = model(**inputs_a).last_hidden_state
        v = model(**inputs_b).last_hidden_state
    u = mean_pool(u, attention_a).cpu().numpy().reshape(-1)
    v = mean_pool(v, attention_b).cpu().numpy().reshape(-1)
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]
    return similarity_score

# Streamlit App
st.title("Text Similarity with BERT")

sentence_a = st.text_input("Enter the first sentence:")
sentence_b = st.text_input("Enter the second sentence:")

if st.button("Calculate Similarity"):
    if sentence_a and sentence_b:
        similarity = calculate_similarity(model, tokenizer, sentence_a, sentence_b, device)
        st.write(f"Cosine Similarity: {similarity:.4f}")
    else:
        st.write("Please enter both sentences to calculate similarity.")

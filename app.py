import streamlit as st
import torch
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_model import BERT, mean_pool

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer
model_path = 'bert.pt'
params, state = torch.load(model_path, map_location=device)
model = BERT(**params, device=device).to(device)
model.load_state_dict(state)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    inputs_a = tokenizer(sentence_a, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    inputs_b = tokenizer(sentence_b, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    inputs_ids_a = inputs_a['input_ids']
    attention_a = inputs_a['attention_mask']
    inputs_ids_b = inputs_b['input_ids']
    attention_b = inputs_b['attention_mask']
    segment_ids = torch.zeros(1, 128, dtype=torch.int32).to(device)

    u = model.get_last_hidden_state(inputs_ids_a, segment_ids)
    v = model.get_last_hidden_state(inputs_ids_b, segment_ids)
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)
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

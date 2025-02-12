import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and tokenizer from Hugging Face
model_name = 'roberta-large-mnli'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def classify_nli(model, tokenizer, sentence_a, sentence_b, device):
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    
    # Mapping indices to NLI labels
    labels = ['contradiction', 'neutral', 'entailment']
    result = {labels[i]: probabilities[i].item() for i in range(len(labels))}
    
    return result

# Streamlit App
st.title("NLI with BERT")

# Inputs for sentences
sentence_a = st.text_input("Enter the first sentence:")
sentence_b = st.text_input("Enter the second sentence:")

# Ensure the button only works when both sentences are provided
if sentence_a and sentence_b:
    if st.button("Classify NLI"):
        # If both sentences are available, classify NLI
        result = classify_nli(model, tokenizer, sentence_a, sentence_b, device)
        st.write(f"Entailment: {result['entailment']:.4f}")
        st.write(f"Neutral: {result['neutral']:.4f}")
        st.write(f"Contradiction: {result['contradiction']:.4f}")
else:
    st.write("Please enter both sentences to classify NLI.")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gdown
import os

st.title("Deteksi Komentar Cyberbullying")

model_path = "cyberbullying_model/model.safetensors"
tokenizer_path = "cyberbullying_tokenizer"
model_dir = "cyberbullying_model"

if not os.path.exists(model_path):
    st.info("Mengunduh model...")
    os.makedirs(model_dir, exist_ok=True)
    url = "https://drive.google.com/file/d/1Z6V28h6nKI0WSBA-hbEx4QnPsk4ig4r"
    gdown.download(url, model_path, quiet=False)

@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_area("**Masukkan komentar:**")

if st.button("**Deteksi**") and user_input.strip():
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    label = "Cyberbullying 😡" if pred == 0 else "Bukan Cyberbullying 😊"
    st.divider()
    st.write("**Hasil Deteksi:**")
    if pred == 1:
        st.success(f"### {label} \n(**Probabilitas: {confidence:.2f}**)")
    else:
        st.warning(f"### {label} \n(**Probabilitas: {confidence:.2f}**)")

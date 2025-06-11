# # streamlit_app.py
# import streamlit as st
# import requests

# st.title("Deteksi Komentar Cyberbullying")
# user_input = st.text_area("Masukkan komentar:")

# if st.button("Deteksi"):
#     response = requests.post("http://127.0.0.1:5000/predict", json={"text": user_input})
#     result = response.json()
#     label = "Cyberbullying 😡" if result['prediction'] == 0 else "Normal 😊"
#     st.write(f"**Hasil Deteksi:** {label} (Probabilitas: {result['probability']:.2f})")

# streamlit_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gdown
import os

st.title("Deteksi Komentar Cyberbullying")

# ====== 1. Download model jika belum ada ======
model_path = "cyberbullying_model/model.safetensors"
tokenizer_path = "cyberbullying_tokenizer"
model_dir = "cyberbullying_model"

if not os.path.exists(model_path):
    st.info("Mengunduh model...")
    os.makedirs(model_dir, exist_ok=True)
    url = "https://drive.google.com/uc?id=1Rgqp7lmibftxEe9tML9lcv6KpmAHPBe2"
    gdown.download(url, model_path, quiet=False)

# ====== 2. Load tokenizer dan model ======
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer, model

tokenizer, model = load_model()

# ====== 3. Input dari pengguna ======
user_input = st.text_area("Masukkan komentar:")

# ====== 4. Prediksi ======
if st.button("Deteksi") and user_input.strip():
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    label = "Cyberbullying 😡" if pred == 0 else "Bukan Cyberbullying 😊"
    st.success(f"**Hasil Deteksi:** {label} (Probabilitas: {confidence:.2f})")

# streamlit_app.py
import streamlit as st
import requests

st.title("Deteksi Komentar Cyberbullying")
user_input = st.text_area("Masukkan komentar:")

if st.button("Deteksi"):
    response = requests.post("http://127.0.0.1:5000/predict", json={"text": user_input})
    result = response.json()
    label = "Cyberbullying 😡" if result['prediction'] == 0 else "Normal 😊"
    st.write(f"**Hasil Deteksi:** {label} (Probabilitas: {result['probability']:.2f})")

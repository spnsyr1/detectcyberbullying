# app.py (Flask backend)
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

model = AutoModelForSequenceClassification.from_pretrained("cyberbullying_model")
tokenizer = AutoTokenizer.from_pretrained("cyberbullying_tokenizer")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    inputs = tokenizer(data['text'], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs).item()
    return jsonify({'prediction': pred, 'probability': probs[0][pred].item()})

if __name__ == '__main__':
    app.run(debug=True)

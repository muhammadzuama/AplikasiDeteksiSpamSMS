from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import re
from collections import defaultdict
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# === Model class ===
class SpamDetection(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SpamDetection, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]
        x = self.dropout(h_n)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# === Preprocessing function ===
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    words = text.split()
    text = " ".join([word for word in words if word not in stopwords])

    return text

# === Vocabulary (rebuild dari dataset) ===
data = pd.read_csv('../model/spam-2.csv') ## Sesuaikan dengan lokasi datasetnya
data['Pesan'] = data['Pesan'].apply(preprocessing)
tokenized = [x.split() for x in data['Pesan'].tolist()]
word2idx = defaultdict(lambda: len(word2idx))
word2idx['<PAD>']
for tokens in tokenized:
    for token in tokens:
        word2idx[token]
word2idx = dict(word2idx)

vocab_size = len(word2idx)
max_len = 100

# === Load trained model ===
model = SpamDetection(vocab_size, embed_dim=120, hidden_dim=120)
model.load_state_dict(torch.load('../model/model.pt', map_location=torch.device('cpu'))) ## Sesuaikan dengan lokasi modelnya
model.eval()

# === Flask app ===
app = Flask(__name__)

def predict(text):
    text = preprocessing(text)
    tokens = text.split()
    ids = [word2idx.get(word, 0) for word in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    input_tensor = torch.tensor([ids])

    with torch.no_grad():
        output = model(input_tensor)
        pred = (output > 0.5).float().item()

    label = "Spam" if pred == 1.0 else "Bukan Spam"
    return {"prediction": label, "confidence": float(output.item())}

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Input JSON harus memiliki key 'text'"}), 400

    text = data["text"]
    result = predict(text)
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Deteksi Spam aktif. Gunakan endpoint POST /predict"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

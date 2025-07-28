Berikut adalah README yang lengkap dan profesional untuk proyek Flask API deteksi spam berbasis model PyTorch dan preprocessing Bahasa Indonesia:

---

# 🛡️ Spam Detection API (Bahasa Indonesia)

API ini mendeteksi apakah sebuah pesan teks berbahasa Indonesia tergolong **Spam** atau **Bukan Spam**, menggunakan model berbasis **PyTorch LSTM** yang dilatih dengan data lokal. API ini dibuat dengan **Flask** dan dapat digunakan untuk integrasi ke aplikasi Flutter, web, atau sistem lainnya.

---

## 🚀 Fitur Utama

* 🧠 **Model LSTM** berbasis PyTorch untuk klasifikasi teks.
* 🔤 **Preprocessing teks Bahasa Indonesia** (lowercase, hapus angka, tanda baca, dan stopword).
* 📊 **Prediksi dengan probabilitas** (confidence score).
* 🌐 **Flask API + CORS** untuk akses dari frontend/web/mobile.
* 📥 Endpoint REST: `POST /predict`

---

## 📦 Dependensi Utama

| Library    | Versi yang Direkomendasikan |
| ---------- | --------------------------- |
| Python     | 3.9+                        |
| Flask      | 2.x                         |
| torch      | 2.1.0                       |
| pandas     | 2.2.3                       |
| numpy      | 1.24.3                      |
| sastrawi   | Terbaru                     |
| flask-cors | Terbaru                     |

---

## 📁 Struktur Proyek

```
.
├── app.py                   # Script utama Flask API
├── ../model/
│   ├── model.pt             # File model PyTorch hasil pelatihan
│   └── spam-2.csv           # Dataset pelatihan (digunakan untuk vocab)
```

> **Catatan:** Ubah path `../model/` jika struktur proyek Anda berbeda.

---

## 🔧 Instalasi

### 1. Clone & Masuk ke Folder Proyek

```bash
git clone https://github.com/username/spam-detection-api.git
cd spam-detection-api
```

### 2. Buat & Aktifkan Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

> Jika tidak ada `requirements.txt`, gunakan:

```bash
pip install flask torch pandas numpy sastrawi flask-cors
```

---

## ▶️ Menjalankan API

```bash
python app.py
```

API akan aktif di:
📍 `http://localhost:5000`

---

## 📨 Cara Menggunakan API

### ✅ Endpoint: `POST /predict`

**Request Body:**

```json
{
  "text": "Selamat! Anda mendapatkan hadiah undian gratis!"
}
```

**Response:**

```json
{
  "prediction": "Spam",
  "confidence": 0.9632
}
```

### 🔄 Endpoint: `GET /`

Mengecek apakah API aktif.

---

## 🧠 Tentang Model

* Model dilatih menggunakan arsitektur **LSTM**.
* Input diproses dengan **tokenisasi manual** dan **stopword removal (Sastrawi)**.
* Model memetakan kata ke indeks (`word2idx`) dan mem-padding input hingga panjang tetap (`max_len = 100`).
* Output adalah nilai antara 0–1 yang menunjukkan probabilitas pesan tergolong spam.

---

## 🧪 Contoh Request via `curl`

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Gratis pulsa hanya hari ini klik link ini!"}'
```

---

## 🛠 Tips Debug

* Jika muncul error terkait **CSV** atau **model path**, pastikan file `spam-2.csv` dan `model.pt` tersedia di jalur `../model/`.
* Pastikan juga model dilatih dengan vocabulary dan struktur preprocessing yang sama.

---

## 📜 Lisensi

Lisensi bebas digunakan untuk riset dan pengembangan non-komersial. Untuk penggunaan komersial, silakan hubungi pengembang.

---

## 👨‍💻 Kontributor

* Muhammad Zuama Al Amin – *Backend Developer & ML Engineer*

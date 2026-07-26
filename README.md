# Deploy ke Streamlit Community Cloud

Struktur folder ini sudah siap dipakai. Isinya:

```
ocr_diet_streamlit/
├── app.py            # Aplikasi Streamlit (hasil OCR + NER + Rule Engine)
├── requirements.txt  # Dependency Python
├── packages.txt      # Dependency sistem (apt) untuk OpenCV/PaddleOCR
├── model-best/       # <-- WAJIB kamu tambahkan sendiri (lihat langkah 1)
└── README.md
```

## Langkah 1 — Siapkan model NER hasil training

Notebook kamu (Bagian 12–14) menghasilkan folder `model-best` (dan `model-last`),
lalu dikemas jadi `model-best.zip` yang otomatis ter-download di Colab.

1. Ambil `model-best.zip` dari hasil training tersebut.
2. Ekstrak, lalu taruh isinya di folder `model-best/` **persis di sebelah `app.py`**
   (bukan `app.py` di dalam `model-best/`).
3. Ukuran folder model spaCy NER biasanya kecil (beberapa MB), jadi aman
   di-commit langsung ke GitHub.

Kalau kamu belum sempat training ulang, jalankan dulu Bagian 8.5–14 di
notebook Colab untuk menghasilkan `model-best.zip`.

## Langkah 2 — Push ke GitHub

```bash
cd ocr_diet_streamlit
git init
git add app.py requirements.txt packages.txt model-best
git commit -m "OCR label gizi diabetes - Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

> Kalau `model-best/` ternyata cukup besar (>100 MB per file), GitHub akan
> menolak push biasa — pakai [Git LFS](https://git-lfs.com/) untuk file itu,
> atau upload model ke Hugging Face Hub / Google Drive lalu download saat
> `app.py` pertama kali start (tambahkan kode download di awal `load_ner_model`).

## Langkah 3 — Deploy di Streamlit Community Cloud

1. Buka https://share.streamlit.io → **New app**.
2. Pilih repo, branch `main`, dan file utama `app.py`.
3. Klik **Deploy**.
4. Tunggu proses build (instalasi `paddlepaddle` + `paddleocr` bisa memakan
   waktu 5–10 menit karena ukurannya besar).

## ⚠️ Catatan penting soal resource

- Streamlit Community Cloud (gratis) membatasi tiap app di sekitar **1 CPU
  dan ~1 GB RAM**. `paddlepaddle` + `paddleocr` + model spaCy tergolong
  cukup berat, jadi ada risiko app crash/OOM (`Oh no. Error running app.`)
  terutama saat memproses gambar resolusi besar.
- Kalau ini terjadi, opsi yang bisa dicoba:
  - Turunkan `max_side` pada `resize_if_too_large()` di `app.py` (mis. dari
    1800 ke 1000–1200 px) supaya OCR lebih ringan.
  - Pertimbangkan platform dengan resource lebih besar untuk beban ML
    seperti ini, misalnya **Hugging Face Spaces** (mendukung Streamlit juga,
    dengan pilihan CPU/RAM lebih besar bahkan di tier gratis) atau
    **Streamlit Community Cloud versi berbayar / self-host** di VM sendiri.
- Gunakan `opencv-python-headless` (bukan `opencv-python`) — sudah diatur di
  `requirements.txt` — karena versi biasa butuh library GUI yang tidak ada
  di server headless dan sering menyebabkan `ImportError: libGL.so.1`.

## Yang TIDAK ikut di app ini (sengaja)

Bagian training (spaCy NER training, konversi dataset anotasi ke `DocBin`,
`spacy train`, evaluasi model) dari notebook **tidak** disertakan di
`app.py` ini — itu proses satu kali yang seharusnya kamu jalankan di
Colab/lokal, bukan tiap kali app di-deploy. `app.py` di sini murni tahap
**inference**: upload gambar → OCR → NER → Rule Engine.

# SpotCheck

Aplikasi web klasifikasi citra kulit: membedakan **eczema** dan **tinea**
(kurap) dari satu foto, sekaligus menyajikan halaman edukasi untuk kedua
kondisi tersebut.

Aplikasi ini adalah deliverable web dari Proyek Ilmiah / skripsi. Modelnya
berupa CNN residual yang dilatih dari nol dan sudah selesai lebih dulu;
repositori ini membangun aplikasi web di sekitarnya.

> ⚠️ **Bukan alat diagnosis medis.** SpotCheck adalah alat bantu edukatif, bukan
> dokter. Hasilnya tidak boleh dipakai sebagai dasar pengobatan. Temui dokter
> atau dokter spesialis kulit untuk memastikan kondisi apa pun.

---

## Kebutuhan

- Python 3.12
- ~1 GB ruang disk (TensorFlow + model)
- Tidak butuh GPU — inferensi memakai `tensorflow-cpu`

## Menjalankan di lokal

```powershell
# 1. Buat virtual environment
python -m venv venv

# 2. Pasang dependensi
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Siapkan konfigurasi
copy .env.example .env
# lalu isi SECRET_KEY di .env dengan nilai acak:
#   python -c "import secrets; print(secrets.token_hex(32))"

# 4. Jalankan
.\venv\Scripts\python.exe run.py
```

Buka **http://127.0.0.1:5000**. Hentikan server dengan `Ctrl + C`.

> Perintah di atas memanggil `python.exe` di dalam venv secara langsung, jadi
> venv tidak perlu diaktifkan. Ini menghindari galat *execution policy*
> PowerShell yang sering muncul saat menjalankan `Activate.ps1`.

## Menjalankan pengujian

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest
```

Uji yang tersedia mencakup geometri letterbox, kontrak keluaran `/predict`,
validasi upload, dan — yang paling penting — pembuktian bahwa preprocessing
aplikasi **identik** dengan pipeline saat training.

---

## Konfigurasi

Dibaca dari `.env` (lihat `.env.example`). Kunci yang dikosongkan akan memakai
nilai default.

| Variabel | Default | Keterangan |
|---|---|---|
| `SECRET_KEY` | — | Kunci rahasia Flask. Wajib diisi untuk produksi. |
| `FLASK_ENV` | `development` | `development` atau `production`. |
| `MODEL_PATH` | `app/ml/model_final_best.keras` | Lokasi berkas model. |
| `MAX_CONTENT_LENGTH` | `8388608` (8 MB) | Batas ukuran upload. |

---

## Struktur proyek

```
SpotCheck-Flask/
├── app/
│   ├── __init__.py          # create_app() — application factory
│   ├── routes.py            # "/" (halaman) dan "/predict" (inferensi)
│   ├── ml/
│   │   ├── inference.py     # muat model sekali + letterbox + preprocess + predict
│   │   └── model_final_best.keras
│   ├── static/{css,js,img}  # aset hasil ekstraksi dari prototype
│   └── templates/           # base.html + index.html
├── tests/                   # pytest: preprocessing, route, validasi
├── config.py                # Config / DevConfig / ProdConfig
├── run.py                   # entry point pengembangan
└── requirements.txt         # produksi (requirements-dev.txt untuk pengujian)
```

Seluruh halaman edukasi bersifat statis dan navigasinya dilakukan di sisi klien,
sehingga aplikasi menyajikan **satu halaman** (`index.html`) dengan empat bagian:
Home & Scan, Eczema, Tinea, dan About Model.

---

## Model

| | |
|---|---|
| Arsitektur | `Final_Residual_CNN` — CNN residual, dilatih dari nol |
| Parameter | 747.713 (~748K) |
| Input | 224 × 224 × 3, RGB, nilai piksel 0–1 |
| Output | satu nilai sigmoid = P(tinea) |
| Kelas | `0 = eczema`, `1 = tinea` |
| Ambang | `p >= 0.5` → Tinea; `p < 0.5` → Eczema |
| Dataset | 2.020 citra bersih (1.037 eczema / 983 tinea) |
| Performa | akurasi 83,2% · ROC-AUC 89,7% pada test set |

Model dimuat **sekali** saat startup (`create_app()`), bukan per request.

### Preprocessing — harus sama persis dengan training

Model dilatih memakai **letterbox resize** (skala proporsional + padding hitam
di tengah) lalu normalisasi `/255`. Model **tidak** memiliki layer `Rescaling`
di dalamnya, sehingga pembagian 255 wajib dilakukan di `inference.py`.

Bila langkah ini menyimpang dari cara model dilatih, prediksi akan meleset
**tanpa memunculkan error apa pun** — karena itu kesamaannya dikunci oleh uji
otomatis (`tests/test_inference.py::test_preprocess_matches_training_pipeline`)
yang membandingkan keluaran `preprocess()` dengan replikasi pipeline notebook,
dan mensyaratkan keduanya identik.

---

## API

### `POST /predict`

Menerima `multipart/form-data` dengan field `image` (satu berkas JPG atau PNG).

**Berhasil — `200`**

```json
{
  "verdict": "Eczema",
  "confidence": 87,
  "eczema_pct": 87,
  "tinea_pct": 13,
  "probability_tinea": 0.1267311573028564
}
```

**Gagal — `400` / `413` / `500`**

```json
{ "error": "Berkas ini bukan citra yang valid." }
```

Citra diproses di memori untuk prediksi dan **tidak disimpan** ke disk.

---

## Deployment

Aplikasi dikemas sebagai image Docker yang menjalankan **gunicorn**. Target yang
disarankan adalah **Render** — gratis, tanpa kartu kredit, dan mendukung Docker.

### Kebutuhan sumber daya (hasil pengukuran)

| | |
|---|---|
| RAM saat berjalan | **~303 MB** per worker (TensorFlow + model), stabil setelah puluhan prediksi |
| TensorFlow di disk | ~1,3 GB — hidup di dalam image Docker, bukan di kuota disk host |
| Ukuran image | ~1,5–2 GB (batas Render: 10 GB) |
| Waktu prediksi | ~67 ms pada CPU desktop |

Karena itu `Dockerfile` memakai **1 worker** secara default (`WEB_CONCURRENCY=1`):
dua worker butuh ~600 MB dan tidak akan muat di host 512 MB. Di host bermemori
lega, naikkan lewat environment variable `WEB_CONCURRENCY`.

Port dibaca dari environment variable `PORT` (default 7860), sehingga image yang
sama jalan di Render (yang menyuntikkan `PORT` sendiri) maupun platform lain
tanpa perlu mengubah `Dockerfile`.

### Deploy ke Render (gratis)

Render menarik kode dari repositori Git, jadi proyek perlu ada di GitHub dulu.

**1. Buat repositori kosong** di https://github.com/new

- *Repository name*: `spotcheck-flask`
- **Jangan** centang "Add a README file" — repo harus kosong

**2. Hubungkan dan push** dari folder proyek:

```powershell
git remote add origin https://github.com/USERNAME/spotcheck-flask.git
git branch -M main
git push -u origin main
```

**3. Buat Web Service** di https://dashboard.render.com/create?type=web

- Hubungkan akun GitHub, lalu pilih repo `spotcheck-flask`
- *Language*: **Docker** (terdeteksi otomatis dari `Dockerfile`)
- *Instance Type*: **Free**
- Sisanya biarkan default — port dan perintah start sudah diatur `Dockerfile`

**4. Tambahkan environment variable** sebelum menekan Create:

| Key | Value |
|---|---|
| `SECRET_KEY` | hasil `python -c "import secrets; print(secrets.token_hex(32))"` |

**5. Klik Create Web Service.** Build pertama makan waktu ~5–15 menit karena
mengunduh dan memasang TensorFlow; ikuti prosesnya di tab **Logs**. Setelah
statusnya **Live**, aplikasi bisa diakses siapa pun lewat URL `.onrender.com`.

### Batasan free tier Render yang perlu diketahui

- **512 MB RAM · 0,1 CPU** — cukup untuk aplikasi ini (~303 MB), tapi prediksi
  jadi ~0,5–2 detik, bukan 67 ms.
- **Tidur setelah 15 menit** tanpa pengunjung. Kunjungan berikutnya menunggu
  ~1 menit sampai container bangun dan TensorFlow selesai dimuat. Untuk demo
  atau sidang: buka URL-nya beberapa menit sebelum mulai.
- **750 jam/bulan** waktu instance.

### Menguji image di lokal (opsional, butuh Docker)

```powershell
docker build -t spotcheck .
docker run --rm -p 7860:7860 -e SECRET_KEY=uji-lokal spotcheck
```

Lalu buka http://127.0.0.1:7860.

### Alternatif: Hugging Face Spaces (kini butuh langganan PRO)

Hugging Face memindahkan Docker Space ke belakang paywall: *"Static Spaces are
free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic
requires a PRO subscription."* Bila berlangganan PRO (2 vCPU · 16 GB RAM,
jauh lebih responsif), `Dockerfile` ini jalan tanpa perubahan — tambahkan
frontmatter berikut di baris paling atas `README.md`:

```yaml
---
title: SpotCheck
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

---

## Catatan pengembangan

Desain diambil dari `spotcheck-prototype.html` dan dipertahankan identik; berkas
prototype itu sendiri tidak diubah dan tetap menjadi rujukan desain. Dua
penyimpangan yang disengaja terhadap prototype, keduanya terdokumentasi di
`CLAUDE.md`:

1. **`.bar-fill{display:block;}`** — perbaikan bug. Pada prototype, `.bar-fill`
   adalah `<span>` tanpa properti `display` sehingga menjadi elemen inline, dan
   elemen inline mengabaikan `width`/`height`. Akibatnya bar hasil tidak pernah
   tampil sama sekali.
2. **Alur scan di dropzone** — preview foto, animasi garis pindai saat
   menganalisis (ditahan minimal 1,2 detik agar sempat terlihat), foto tampil
   utuh setelah hasil keluar, serta tombol "Scan another photo" dan petunjuk
   hover untuk mencoba foto lain. Dropzone dalam keadaan idle tidak berubah.

Foto pada halaman edukasi semula tertanam sebagai base64 di dalam prototype dan
telah didekode menjadi `app/static/img/*.jpg` (byte-nya identik, diverifikasi
dengan sha256), sehingga `index.html` tetap ringan dan bisa dibaca.

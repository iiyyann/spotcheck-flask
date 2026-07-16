---
title: SpotCheck
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SpotCheck

Aplikasi web klasifikasi citra kulit: membedakan **eczema** dan **tinea**
(kurap) dari satu foto, sekaligus menyajikan halaman edukasi untuk kedua
kondisi tersebut.

<!-- Blok YAML di atas adalah konfigurasi Hugging Face Spaces (wajib ada di
     README.md sebuah Space). Di GitHub blok itu tampil sebagai tabel kecil;
     abaikan saja. Jangan dihapus bila aplikasi di-deploy ke Spaces. -->


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
disarankan adalah **Hugging Face Spaces (Docker Space)** — gratis, ramah beban
kerja ML, dan memorinya lega (TensorFlow + model butuh beberapa ratus MB saat
dimuat).

`Dockerfile` sudah disiapkan untuk Spaces: melayani di **port 7860** dan berjalan
sebagai user non-root ber-UID 1000, sesuai yang diharapkan platform itu.

### Menguji image di lokal (opsional, butuh Docker)

```powershell
docker build -t spotcheck .
docker run --rm -p 7860:7860 -e SECRET_KEY=uji-lokal spotcheck
```

Lalu buka http://127.0.0.1:7860.

### Deploy ke Hugging Face Spaces

Frontmatter Spaces sudah tertanam di bagian atas README ini, jadi tidak ada yang
perlu digabungkan manual.

**1. Buat Space** di https://huggingface.co/new-space

- *Space name*: `spotcheck` (bebas)
- *License*: `mit` (bebas)
- *SDK*: **Docker** → template **Blank**
- *Hardware*: **CPU basic · free**
- *Visibility*: Public

**2. Clone repo Space** ke folder mana pun di luar folder proyek:

```powershell
cd $HOME\Documents
git clone https://huggingface.co/spaces/USERNAME/spotcheck spotcheck-space
```

Bila diminta login, pakai username Hugging Face dan **Access Token** sebagai
password — buat token di https://huggingface.co/settings/tokens dengan peran
*write*.

**3. Salin berkas proyek** ke folder Space. Perintah berikut menyalin semuanya
kecuali yang tidak boleh ikut (`venv/` yang berukuran ratusan MB, `.env` yang
berisi kunci rahasia, cache Python, dan folder `.git` milik Space):

```powershell
robocopy "$HOME\Proyek PI\SpotCheck-Flask" "$HOME\Documents\spotcheck-space" /E `
  /XD venv .git __pycache__ .pytest_cache .vscode .idea `
  /XF .env
```

> `robocopy` menghasilkan exit code 1 saat berhasil menyalin — itu **normal**,
> bukan error.

**4. Set `SECRET_KEY`** di halaman Space: *Settings → Variables and secrets →
New secret*.

- Name: `SECRET_KEY`
- Value: hasil dari `python -c "import secrets; print(secrets.token_hex(32))"`

**5. Push:**

```powershell
cd $HOME\Documents\spotcheck-space
git add .
git commit -m "Deploy SpotCheck"
git push
```

**6. Tunggu build.** Space otomatis mem-build image lalu menjalankannya. Build
pertama makan waktu beberapa menit karena mengunduh TensorFlow; ikuti di tab
**Logs** pada halaman Space. Setelah statusnya *Running*, aplikasi bisa diakses
siapa pun lewat URL Space tersebut.

Berkas model berukuran 8,7 MB sehingga **tidak** memerlukan Git LFS (ambang
batas Hugging Face adalah 10 MB).

### Alternatif: Render

`Dockerfile` yang sama bisa dipakai di Render (*New → Web Service → Docker*).
Ubah port pada `--bind` dan `EXPOSE` menjadi `10000`, atau baca dari environment
variable `PORT` yang disediakan Render.

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

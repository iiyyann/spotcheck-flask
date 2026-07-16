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
| `EAGER_LOAD_MODEL` | `1` | `1` = muat model saat startup. **Wajib `0`** di server yang mem-fork (Passenger/cPanel) — lihat catatan fork di bagian Deployment. |

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

Model dimuat **sekali** lalu dipakai ulang untuk setiap request — tidak pernah
dimuat per request. Waktu pemuatannya bergantung `EAGER_LOAD_MODEL`: saat startup
(default) atau saat request pertama. Lihat *Catatan penting: TensorFlow dan fork()*
di bagian Deployment.

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

Aplikasi ini **sudah ter-deploy** dan dapat diakses publik di:

**https://spotcheck.my.id** — ArenHost, Business Hosting Advance (cPanel), Rp25.000/bulan

Sertifikat SSL diterbitkan Let's Encrypt lewat AutoSSL cPanel dan diperpanjang
otomatis. *Force HTTPS Redirect* aktif, sehingga kunjungan `http://` dialihkan
ke `https://`.

Deployment memakai fitur **Setup Python App** (Phusion Passenger) di cPanel.
`Dockerfile` yang tersedia di repositori ini dipakai untuk platform berbasis
kontainer dan tidak dipakai di shared hosting.

### Kebutuhan sumber daya (hasil pengukuran)

| | |
|---|---|
| RAM saat berjalan | **~303 MB** per proses (TensorFlow + model), stabil setelah puluhan prediksi |
| TensorFlow di disk | ~1,3 GB — total dependensi ~1,5 GB |
| Berkas model | 8,7 MB |
| Waktu prediksi | ~67 ms (CPU desktop) · **~0,5 detik** (server produksi) |
| RAM tersedia di host | 4 GB (dikonfirmasi ArenHost) |

Angka RAM inilah yang menentukan kelayakan sebuah host, bukan ukuran disk.
TensorFlow besar di disk (1,3 GB) tetapi hanya memakai ~303 MB memori saat
berjalan — dua hal yang sering tertukar saat memilih hosting.

### Langkah deployment di cPanel

1. **Setup Python App → Create Application**

   | Kolom | Nilai |
   |---|---|
   | Python version | `3.12.13` (samakan dengan versi pengembangan) |
   | Application root | `spotcheck` |
   | Application URL | `spotcheck.my.id` (path dikosongkan) |
   | Application startup file | `passenger_wsgi.py` |
   | Application Entry point | `application` |

2. **Ambil kode lewat Terminal cPanel.** cPanel sudah membuat
   `passenger_wsgi.py` contoh di folder aplikasi, sehingga `git clone` menolak
   menulis ke folder yang tidak kosong. Karena itu clone lewat folder sementara:

   ```bash
   cd ~
   git clone https://github.com/iiyyann/spotcheck-flask.git spotcheck-tmp
   cp -rf spotcheck-tmp/. spotcheck/
   rm -rf spotcheck-tmp
   ```

3. **Pasang dependensi.** Pada halaman Setup Python App, bagian *Configuration
   files*, tambahkan `requirements.txt` lalu klik **Run Pip Install**
   (~5–15 menit; TensorFlow 1,3 GB).

   > cPanel mungkin menampilkan error *"check availability of application has
   > failed ... content type"*. Itu **gangguan kosmetik**: pemeriksaan kesehatan
   > cPanel mengeluh karena respons aplikasi berubah menjadi
   > `text/html; charset=utf-8`. Instalasinya sendiri tetap berjalan — verifikasi
   > dengan `pip list`.

4. **Tambahkan environment variable** `SECRET_KEY` → **SAVE** → **RESTART**.

5. **Verifikasi** dari terminal, di dalam virtualenv:

   ```bash
   source /home/spotchec/virtualenv/spotcheck/3.12/bin/activate && cd ~/spotcheck
   python -c "from app.ml import inference; inference.load_model('app/ml/model_final_best.keras'); print(inference.predict('app/static/img/tinea.jpg'))"
   ```

### SSL / HTTPS

AutoSSL berjalan terjadwal dan menerbitkan sertifikat Let's Encrypt secara
otomatis, **asalkan DNS domain sudah mengarah ke server**. Bila AutoSSL sempat
berjalan sebelum DNS propagasi, statusnya akan tertulis:

> `"spotcheck.my.id" is unmanaged. Verify this domain's registration and
> authoritative nameserver configuration.`

Itu bukan kegagalan permanen — cukup tunggu jadwal AutoSSL berikutnya setelah DNS
benar, atau minta support menjalankannya ulang. Statusnya dapat dilihat di
**SSL/TLS Certificates → Status**.

Setelah sertifikat terbit (status berubah dari *Self-signed* menjadi *AutoSSL
Domain Validated*), nyalakan **Force HTTPS Redirect** di cPanel → **Domains**.
Jangan dinyalakan sebelum sertifikat terbit: pengunjung akan dipaksa ke HTTPS
yang sertifikatnya masih self-signed dan melihat peringatan keamanan.

### Memperbarui aplikasi setelah ada perubahan

```bash
# di komputer lokal
pytest && git add -A && git commit -m "..." && git push

# di Terminal cPanel
cd ~/spotcheck && git pull
```

Lalu klik **RESTART** di Setup Python App. **Restart wajib** — Passenger menahan
aplikasi lama di memori, sehingga tanpa restart kode baru tidak pernah dijalankan.

### Catatan penting: TensorFlow dan fork()

Passenger memuat aplikasi di proses induk lalu **mem-fork** proses pekerja.
TensorFlow **tidak aman terhadap fork**: model yang dimuat sebelum fork akan
membuat `model.predict()` **menggantung selamanya** di proses anak — tanpa pesan
error, dan tanpa gejala lain (route dan validasi tetap berjalan normal).

Karena itu `passenger_wsgi.py` menyetel `EAGER_LOAD_MODEL=0`, sehingga model
dimuat saat request pertama, di dalam proses pekerja, setelah fork. Konsekuensinya
request pertama pada proses baru butuh ~8–20 detik; sesudah itu ~0,5 detik.

Alasan yang sama membuat `Dockerfile` tidak memakai `--preload` gunicorn: setiap
worker memuat modelnya sendiri.

### Sebelum sesi demo atau pengujian

Buka situs dan lakukan **satu prediksi** beberapa menit sebelum sesi dimulai.
Passenger mematikan proses yang lama menganggur; pengunjung pertama sesudahnya
harus menunggu model dimuat ulang.

### Alternatif yang dipertimbangkan

| Platform | Status |
|---|---|
| **Hugging Face Spaces** | Docker Space kini butuh langganan **PRO** — tidak lagi gratis |
| **Render** | Free tier (512 MB) muat untuk aplikasi ini, tetapi **meminta kartu kredit** |
| **Google Cloud Run / Fly.io** | Wajib kartu kredit |
| **ArenHost (dipilih)** | Rp25.000/bulan, bayar transfer/QRIS, RAM 4 GB |

`Dockerfile` tetap dipertahankan agar aplikasi bisa dipindahkan ke platform
kontainer mana pun tanpa perubahan kode: port dibaca dari `PORT`
(default 7860), jumlah worker dari `WEB_CONCURRENCY` (default 1, ~303 MB).

### Menguji image Docker di lokal (opsional)

```powershell
docker build -t spotcheck .
docker run --rm -p 7860:7860 -e SECRET_KEY=uji-lokal spotcheck
```

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

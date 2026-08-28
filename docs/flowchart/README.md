# Flowchart aplikasi SpotCheck

Diagram alir standar (hitam putih) untuk dilampirkan ke laporan. Setiap gambar
tersedia dalam dua format:

- `*.svg` — vektor, tidak pecah saat diperbesar. Gunakan ini untuk Word 2016 ke
  atas dan LaTeX.
- `*.png` — raster 3x (lebar ± 2940 px), untuk perkakas yang belum mendukung SVG.

| Berkas | Isi | Usulan keterangan gambar |
| --- | --- | --- |
| `flowchart-1-alur-utama` | Perjalanan pengguna dari membuka aplikasi sampai hasil klasifikasi tampil | Gambar 3.x Alur utama aplikasi SpotCheck |
| `flowchart-2-endpoint-predict` | Validasi unggahan dan penentuan kode respons pada `POST /predict` | Gambar 3.x Alur penanganan permintaan pada endpoint `/predict` |
| `flowchart-3-prapemrosesan-model` | Prapemrosesan citra dan inferensi model | Gambar 3.x Alur prapemrosesan citra dan inferensi model |
| `flowchart-4-inisialisasi-aplikasi` | Urutan penyiapan aplikasi saat startup | Gambar 3.x Alur inisialisasi aplikasi |

## Simbol yang dipakai

| Simbol | Arti |
| --- | --- |
| Persegi panjang sudut membulat | Terminator (mulai / selesai) |
| Persegi panjang | Proses |
| Belah ketupat | Keputusan |
| Jajar genjang | Masukan / keluaran |
| Persegi panjang bergaris tepi ganda | Proses terdefinisi (dirinci di gambar lain) |

## Sumber

Isi diagram diturunkan dari kode yang berjalan:
`app/__init__.py`, `app/routes.py`, `app/ml/inference.py`, `config.py`, dan
`app/static/js/main.js`.

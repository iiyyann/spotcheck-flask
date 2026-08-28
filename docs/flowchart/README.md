# Flowchart aplikasi SpotCheck

Diagram alir standar (hitam putih) untuk dilampirkan ke laporan. Setiap gambar
tersedia dalam dua format:

- `*.svg` — vektor, tidak pecah saat diperbesar. Gunakan ini untuk Word 2016 ke
  atas dan LaTeX.
- `*.png` — raster 3x, untuk perkakas yang belum mendukung SVG.

## Berkas yang dipakai

| Berkas (`-revisi`) | Isi | Usulan letak & keterangan gambar |
| --- | --- | --- |
| `flowchart-pemindaian-citra-kulit-revisi` | Perjalanan pengguna dari membuka aplikasi sampai hasil klasifikasi tampil; mencakup validasi ganda (peramban + server) dan opsi memindai foto lain | **3.9.1 a)** — Gambar 3.42 Flowchart Pemindaian Citra Kulit |
| `flowchart-endpoint-predict-revisi` | Validasi unggahan dan penentuan kode respons 200 / 400 / 413 / 500 pada `POST /predict` | **3.9.1 b)** — Gambar 3.43 Flowchart Penanganan Permintaan pada Endpoint /predict |
| `flowchart-prapemrosesan-inferensi-revisi` | Prapemrosesan citra (EXIF, RGB, rotasi, letterbox, normalisasi) dan inferensi model sampai ambang 0,5 | **3.9.1 c)** — Gambar 3.44 Flowchart Prapemrosesan Citra dan Inferensi Model |
| `flowchart-inisialisasi-aplikasi-revisi` | Urutan penyiapan aplikasi saat startup, termasuk pemuatan model segera atau ditunda | **3.9.1 d)** — Gambar 3.45 Flowchart Inisialisasi Aplikasi |
| `struktur-navigasi-aplikasi` | Digambar menyerupai tampilan asli: topbar berisi tombol menu, tiap halaman berupa kotak yang memuat kotak bagiannya, akordion ditandai tanda panah bawah, ditambah lima tautan asosiatif dari halaman utama | **3.9.2** — Gambar 3.46 Struktur Navigasi Aplikasi |

Teks pada Gambar 3.46 memakai bahasa Inggris karena isinya label antarmuka
yang sebenarnya (nama halaman, isi menu, judul navigasi cepat, judul seksi,
dan judul tombol), ditulis lengkap tanpa singkatan dan tanpa em dash.

Nomor gambar di dalam diagram sudah diisi sesuai penomoran Bab 3 saat ini:
gambar terakhir sebelum subbab 3.9 adalah Gambar 3.41 (Kurva ROC), sehingga
keempat flowchart menjadi Gambar 3.42 sampai 3.45 dan diletakkan seluruhnya di
subbab 3.9.1 dengan penomoran a) sampai d). Bila ada gambar lain yang disisipkan
sebelum 3.9.1, nomor di dalam diagram harus dibuat ulang.

## Berkas lama — JANGAN DIPAKAI

`flowchart-1-alur-utama`, `flowchart-2-endpoint-predict`,
`flowchart-3-prapemrosesan-model`, dan `flowchart-4-inisialisasi-aplikasi`
adalah versi sebelum revisi. Semuanya sudah digantikan oleh berkas `-revisi`
di atas dan hanya disimpan sebagai arsip. Perbedaan utamanya:

- alur utama lama tidak punya percabangan setelah validasi di server, sehingga
  unggahan yang ditolak tetap mengalir ke prapemrosesan;
- alur utama lama juga belum menggambarkan opsi memindai foto lain;
- respons 200 lama belum menyebut `probability_dermatophytosis`;
- kotak perhitungan lama hanya memuat rumus tingkat keyakinan, belum rumus
  persentase kedua kelas.

## Ukuran dan penempatan di Word

Ukuran kanvas sengaja dibuat rapat agar tiap gambar muat dalam satu halaman
ketika ditempel selebar teks (sekitar 15 cm), masih menyisakan ruang untuk
caption.

| Gambar | Kanvas (px) | Rasio lebar : tinggi | Tinggi pada lebar 15 cm |
| --- | --- | --- | --- |
| 3.42 Pemindaian Citra Kulit | 1180 x 1332 | 1 : 1,13 | ± 16,9 cm |
| 3.43 Endpoint /predict | 1170 x 1120 | 1 : 0,96 | ± 14,4 cm |
| 3.44 Prapemrosesan dan Inferensi | 1150 x 1168 | 1 : 1,02 | ± 15,2 cm |
| 3.45 Inisialisasi Aplikasi | 1140 x 810 | 1 : 0,71 | ± 10,7 cm |
| 3.46 Struktur Navigasi | 1340 x 1030 | 1 : 0,77 | ± 11,5 cm |

## Ejaan nama kelas

Mengikuti kebiasaan yang sudah konsisten di Bab 3: **prosa memakai ejaan
Indonesia** (dermatofitosis), sedangkan **nama teknis ditulis apa adanya**
(nama direktori, nama halaman antarmuka, dan field JSON). Karena itu:

- Gambar 3.44 memakai *dermatofitosis* - isinya menjelaskan konsep;
- Gambar 3.43 tetap memakai `dermatophytosis_pct` dan
  `probability_dermatophytosis` - itu nama field JSON yang harus persis;
- Gambar 3.42 tetap menulis "Dermatophytosis" pada daftar halaman edukasi,
  karena itu nama halaman yang benar-benar tampil di antarmuka dan akan
  terlihat pada tangkapan layar di subbab 3.9.3.

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

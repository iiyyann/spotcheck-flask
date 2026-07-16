"""Entry point untuk cPanel (Phusion Passenger).

Dipakai saat aplikasi di-deploy ke shared hosting lewat fitur "Setup Python App"
di cPanel. Passenger mengimpor berkas ini dan mencari objek bernama
`application`.

Berkas ini TIDAK dipakai saat menjalankan aplikasi di lokal (pakai run.py) atau
saat deploy dengan Docker/gunicorn (lihat Dockerfile).

Catatan: Passenger membuat proses aplikasi saat request pertama masuk. Impor
TensorFlow plus pemuatan model butuh beberapa detik, sehingga request pertama
setelah aplikasi idle akan terasa lambat. Ini normal.
"""

import os
import sys
from pathlib import Path

# Pastikan folder aplikasi ada di sys.path; Passenger tidak selalu menyetel
# working directory ke folder ini.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Shared hosting memakai CPU biasa tanpa instruksi khusus; matikan log verbose
# TensorFlow agar tidak membanjiri log error cPanel.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Batasi thread TensorFlow. Shared hosting membatasi jumlah proses/thread per
# akun (LVE), dan default TF yang membuka thread sebanyak core fisik server
# bisa menabrak limit itu.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

os.environ.setdefault("FLASK_ENV", "production")

from app import create_app  # noqa: E402

# Nama `application` wajib — inilah yang dicari Passenger.
application = create_app()

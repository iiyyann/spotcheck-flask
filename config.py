"""Konfigurasi aplikasi SpotCheck.

Nilai sensitif dan path dibaca dari environment variable (.env), tidak
di-hardcode. Tiga kelas: Config (dasar), DevConfig, ProdConfig.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Muat .env lebih dulu agar os.environ terisi saat kelas di bawah dievaluasi.
load_dotenv(BASE_DIR / ".env")


# Nilai default SECRET_KEY. Dipakai create_app() untuk mendeteksi bahwa kunci
# belum diganti saat berjalan di produksi.
DEFAULT_SECRET_KEY = "dev-only-secret-change-me"


def _env(name, default):
    """Baca environment variable, anggap string kosong sebagai tidak diisi.

    Perlu karena .env menuliskan kunci tanpa nilai (misal `MODEL_PATH=`) untuk
    menandakan "pakai default"; os.environ.get() akan mengembalikan "" pada
    kasus itu, bukan default-nya.
    """
    value = os.environ.get(name)
    return value if value else default


class Config:
    """Konfigurasi dasar yang dipakai semua environment."""

    SECRET_KEY = _env("SECRET_KEY", DEFAULT_SECRET_KEY)

    # Lokasi model Keras yang sudah dilatih.
    MODEL_PATH = _env("MODEL_PATH", str(BASE_DIR / "app" / "ml" / "model_final_best.keras"))

    # Batas ukuran upload (default 8 MB). Flask menolak request lebih besar
    # dengan error 413 sebelum file sempat dibaca ke memori.
    MAX_CONTENT_LENGTH = int(_env("MAX_CONTENT_LENGTH", 8 * 1024 * 1024))

    # Tipe gambar yang diterima endpoint /predict.
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

    # Muat model saat startup (bukan saat request pertama).
    EAGER_LOAD_MODEL = True


class DevConfig(Config):
    """Konfigurasi untuk pengembangan di mesin lokal."""

    DEBUG = True
    TEMPLATES_AUTO_RELOAD = True


class ProdConfig(Config):
    """Konfigurasi untuk deployment (dijalankan lewat gunicorn)."""

    DEBUG = False
    TESTING = False


# Pemetaan nama -> kelas, dipakai create_app() lewat env FLASK_ENV.
CONFIG_MAP = {
    "development": DevConfig,
    "production": ProdConfig,
}


def get_config(name=None):
    """Kembalikan kelas config sesuai nama; default ke development."""
    key = (name or os.environ.get("FLASK_ENV") or "development").lower()
    return CONFIG_MAP.get(key, DevConfig)

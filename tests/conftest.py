"""Fixture bersama untuk pengujian SpotCheck."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402

# Dua foto milik halaman edukasi dipakai sebagai citra uji yang realistis:
# keduanya foto klinis asli dengan label yang sudah diketahui.
ECZEMA_PHOTO = PROJECT_ROOT / "app" / "static" / "img" / "eczema.jpg"
TINEA_PHOTO = PROJECT_ROOT / "app" / "static" / "img" / "tinea.jpg"


@pytest.fixture(scope="session")
def app():
    """Aplikasi Flask dengan model termuat (sekali untuk seluruh sesi uji)."""
    application = create_app("production")
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    """Test client HTTP."""
    return app.test_client()


@pytest.fixture(scope="session")
def eczema_photo():
    return ECZEMA_PHOTO


@pytest.fixture(scope="session")
def tinea_photo():
    return TINEA_PHOTO


def make_jpeg(size, color=(200, 120, 120)):
    """Buat JPEG polos di memori untuk menguji geometri preprocessing.

    Args:
        size: Tuple (lebar, tinggi).
        color: Warna isian RGB.

    Returns:
        BytesIO berisi data JPEG, siap dibaca PIL.
    """
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=95)
    buf.seek(0)
    return buf

"""Uji preprocessing dan inferensi.

Fokus utama berkas ini: membuktikan bahwa preprocessing aplikasi sama persis
dengan pipeline saat training. Bila uji ini gagal, prediksi aplikasi tidak lagi
bisa dipercaya meskipun aplikasinya tetap berjalan tanpa error.
"""

import numpy as np
import pytest
from PIL import Image

from app.ml import inference
from tests.conftest import make_jpeg


# --------------------------------------------------------------------------
# Letterbox: rasio aspek dipertahankan, padding hitam, citra di tengah
# --------------------------------------------------------------------------

def test_letterbox_returns_target_size():
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    assert inference.letterbox_resize(img, (224, 224)).size == (224, 224)


def test_letterbox_preserves_aspect_ratio():
    """Citra 1000x200 (rasio 5:1) harus tetap 5:1 setelah diskalakan."""
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    arr = np.asarray(inference.letterbox_resize(img, (224, 224)))

    rows = [y for y in range(224) if arr[y].max() > 0]
    cols = [x for x in range(224) if arr[:, x].max() > 0]

    # Skala = min(224/1000, 224/200) = 0.224 -> tinggi 200*0.224 = 44.8 -> int() = 44
    assert len(rows) == 44
    assert len(cols) == 224


def test_letterbox_centers_the_image():
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    arr = np.asarray(inference.letterbox_resize(img, (224, 224)))

    rows = [y for y in range(224) if arr[y].max() > 0]
    padding_atas = rows[0]
    padding_bawah = 223 - rows[-1]
    assert abs(padding_atas - padding_bawah) <= 1


def test_letterbox_pads_with_black():
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    arr = np.asarray(inference.letterbox_resize(img, (224, 224)))
    assert arr[0].max() == 0      # baris paling atas = padding
    assert arr[223].max() == 0    # baris paling bawah = padding


def test_letterbox_square_image_needs_no_padding():
    img = Image.new("RGB", (500, 500), (0, 0, 255))
    arr = np.asarray(inference.letterbox_resize(img, (224, 224)))
    assert all(arr[y].max() > 0 for y in range(224))


# --------------------------------------------------------------------------
# preprocess(): bentuk, tipe, rentang nilai
# --------------------------------------------------------------------------

def test_preprocess_shape_and_dtype(eczema_photo):
    arr = inference.preprocess(eczema_photo)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32


def test_preprocess_normalizes_to_unit_range(eczema_photo):
    arr = inference.preprocess(eczema_photo)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_preprocess_converts_grayscale_to_rgb():
    """Citra grayscale harus tetap menghasilkan 3 kanal."""
    import io

    buf = io.BytesIO()
    Image.new("L", (300, 300), 128).save(buf, "JPEG")
    buf.seek(0)
    assert inference.preprocess(buf).shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image():
    import io

    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        inference.preprocess(io.BytesIO(b"ini jelas bukan citra"))


@pytest.mark.parametrize("size", [(1000, 200), (200, 1000), (500, 500), (50, 37)])
def test_preprocess_handles_various_shapes(size):
    assert inference.preprocess(make_jpeg(size)).shape == (1, 224, 224, 3)


# --------------------------------------------------------------------------
# Uji paling penting: preprocessing identik dengan notebook
# --------------------------------------------------------------------------

@pytest.mark.parametrize("photo", ["eczema", "tinea"])
def test_preprocess_matches_training_pipeline(photo, eczema_photo, tinea_photo):
    """preprocess() harus sama persis dengan SkinDataGenerator saat training.

    Notebook (versi-7, SkinDataGenerator.__getitem__ dengan augment=False):
        img = Image.open(path).convert("RGB")
        img = letterbox_resize(img, IMAGE_SIZE)
        img = img_to_array(img) / 255.0

    Perbedaan satu-satunya di aplikasi adalah pemakaian np.asarray() alih-alih
    img_to_array(); uji ini membuktikan keduanya menghasilkan array identik.
    """
    from tensorflow.keras.utils import img_to_array

    path = eczema_photo if photo == "eczema" else tinea_photo

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = inference.letterbox_resize(img, inference.IMAGE_SIZE)
        referensi = np.expand_dims(img_to_array(img) / 255.0, axis=0)

    hasil = inference.preprocess(path)

    assert hasil.shape == referensi.shape
    assert hasil.dtype == referensi.dtype
    assert np.array_equal(hasil, referensi), "preprocessing menyimpang dari pipeline training"


# --------------------------------------------------------------------------
# predict(): kontrak keluaran
# --------------------------------------------------------------------------

def test_predict_returns_expected_keys(app, eczema_photo):
    hasil = inference.predict(eczema_photo)
    assert set(hasil) == {
        "verdict", "confidence", "eczema_pct", "tinea_pct", "probability_tinea",
    }


def test_predict_percentages_sum_to_100(app, eczema_photo):
    hasil = inference.predict(eczema_photo)
    assert hasil["eczema_pct"] + hasil["tinea_pct"] == 100


def test_predict_confidence_is_the_larger_class(app, tinea_photo):
    hasil = inference.predict(tinea_photo)
    assert hasil["confidence"] == max(hasil["eczema_pct"], hasil["tinea_pct"])
    assert 50 <= hasil["confidence"] <= 100


def test_predict_verdict_follows_threshold(app, eczema_photo, tinea_photo):
    """Verdict harus konsisten dengan ambang 0.5 pada P(tinea)."""
    for path in (eczema_photo, tinea_photo):
        hasil = inference.predict(path)
        tinea_menang = hasil["probability_tinea"] >= inference.THRESHOLD
        assert (hasil["verdict"] == "Tinea") == tinea_menang


def test_predict_probability_within_unit_range(app, eczema_photo):
    assert 0.0 <= inference.predict(eczema_photo)["probability_tinea"] <= 1.0


def test_predict_is_deterministic(app, tinea_photo):
    """Citra yang sama harus menghasilkan probabilitas yang sama persis."""
    a = inference.predict(tinea_photo)["probability_tinea"]
    b = inference.predict(tinea_photo)["probability_tinea"]
    assert a == b


def test_predict_classifies_reference_photos(app, eczema_photo, tinea_photo):
    """Dua foto klinis milik halaman edukasi harus diklasifikasi dengan benar.

    Ini uji kewarasan menyeluruh (preprocessing + model + pemetaan kelas).
    Bila pemetaan 0=eczema / 1=tinea tertukar, uji ini akan gagal.
    """
    assert inference.predict(eczema_photo)["verdict"] == "Eczema"
    assert inference.predict(tinea_photo)["verdict"] == "Tinea"


# --------------------------------------------------------------------------
# Pemuatan model
# --------------------------------------------------------------------------

def test_load_model_with_missing_file_raises_clear_error(tmp_path):
    """Path model yang salah harus memberi FileNotFoundError yang jelas."""
    # Model disimpan di level modul dan dipakai bersama seluruh sesi uji, jadi
    # nilainya wajib dikembalikan agar uji lain tidak ikut kehilangan model.
    model_asli = inference._model
    inference._model = None
    try:
        with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
            inference.load_model(tmp_path / "tidak_ada.keras")
        with pytest.raises(FileNotFoundError):
            inference.load_model("")   # meniru MODEL_PATH kosong di .env
    finally:
        inference._model = model_asli


def test_get_model_before_loading_raises():
    model_asli = inference._model
    inference._model = None
    try:
        with pytest.raises(RuntimeError):
            inference.get_model()
    finally:
        inference._model = model_asli

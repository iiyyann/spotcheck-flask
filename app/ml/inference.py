"""Pemuatan model dan inferensi SpotCheck.

PENTING — preprocessing di modul ini harus sama persis dengan pipeline saat
training (lihat notebook versi-7, kelas SkinDataGenerator): letterbox resize ke
224x224 lalu normalisasi /255. Model tidak punya layer Rescaling di dalamnya,
jadi pembagian 255 memang harus dilakukan di sini. Bila langkah ini berbeda dari
saat training, prediksi akan meleset tanpa memunculkan error apa pun.

Konvensi kelas (dari training): 0 = eczema, 1 = tinea.
Output model adalah satu nilai sigmoid = P(tinea).
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Ukuran input model, sama dengan IMAGE_SIZE di notebook.
IMAGE_SIZE = (224, 224)

# Ambang keputusan; notebook memakai 0.5 saat evaluasi.
THRESHOLD = 0.5

# Model disimpan di level modul agar hanya dimuat sekali, lalu dipakai ulang
# untuk setiap request.
_model = None


def load_model(model_path):
    """Muat model Keras sekali dan simpan di level modul.

    Dipanggil dari create_app() saat startup, bukan saat request pertama, agar
    request pertama tidak menanggung waktu muat model.

    Args:
        model_path: Path ke berkas .keras.

    Returns:
        Model Keras yang sudah dimuat.

    Raises:
        FileNotFoundError: Bila berkas model tidak ada di path tersebut.
    """
    global _model
    if _model is None:
        path = Path(model_path) if model_path else None
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Berkas model tidak ditemukan: {model_path!r}. "
                "Periksa MODEL_PATH di .env atau keberadaan "
                "app/ml/model_final_best.keras."
            )

        # Impor TensorFlow ditunda sampai di sini supaya proses impor modul
        # tetap ringan (TF butuh beberapa detik untuk diimpor).
        import tensorflow as tf

        _model = tf.keras.models.load_model(path)
    return _model


def get_model():
    """Kembalikan model yang sudah dimuat.

    Raises:
        RuntimeError: Bila load_model() belum pernah dipanggil.
    """
    if _model is None:
        raise RuntimeError("Model belum dimuat. Panggil load_model() lebih dulu.")
    return _model


def letterbox_resize(img, target_size, fill_color=(0, 0, 0)):
    """Ubah ukuran citra secara proporsional lalu beri padding di tengah.

    Rasio aspek citra asli dipertahankan (tidak digepengkan); sisa ruang diisi
    warna fill_color. Ini strategi resize yang dipakai saat training.

    Args:
        img: Citra PIL bermode RGB.
        target_size: Tuple (lebar, tinggi) tujuan.
        fill_color: Warna padding, default hitam seperti saat training.

    Returns:
        Citra PIL berukuran target_size.
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized_img = img.resize((new_w, new_h), Image.BILINEAR)

    padded_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))
    return padded_img


def preprocess(image_file):
    """Ubah berkas citra menjadi array siap-prediksi (1, 224, 224, 3).

    Args:
        image_file: Path, objek file, atau stream yang bisa dibaca PIL.

    Returns:
        Array float32 ternormalisasi 0..1 dengan batch berukuran 1.

    Raises:
        PIL.UnidentifiedImageError: Bila berkas bukan citra yang valid.
    """
    img = Image.open(image_file).convert("RGB")     # 1. paksa RGB
    img = letterbox_resize(img, IMAGE_SIZE)         # 2. letterbox ke 224x224
    arr = np.asarray(img, dtype="float32") / 255.0  # 3. normalisasi 0..1
    arr = np.expand_dims(arr, axis=0)               # 4. batch 1 -> (1,224,224,3)
    return arr


def predict(image_file):
    """Jalankan prediksi untuk satu citra.

    Args:
        image_file: Path, objek file, atau stream berisi citra JPG/PNG.

    Returns:
        Dict berisi verdict, confidence, dan persentase tiap kelas, siap
        dikirim sebagai JSON.

    Raises:
        PIL.UnidentifiedImageError: Bila berkas bukan citra yang valid.
        RuntimeError: Bila model belum dimuat.
    """
    model = get_model()
    arr = preprocess(image_file)

    # Output sigmoid tunggal = P(tinea).
    p_tinea = float(model.predict(arr, verbose=0)[0][0])

    is_tinea = p_tinea >= THRESHOLD
    return {
        "verdict": "Tinea" if is_tinea else "Eczema",
        "confidence": round(max(p_tinea, 1 - p_tinea) * 100),
        "eczema_pct": round((1 - p_tinea) * 100),
        "tinea_pct": round(p_tinea * 100),
        # Probabilitas mentah disertakan untuk keperluan debugging/analisis.
        "probability_tinea": p_tinea,
    }

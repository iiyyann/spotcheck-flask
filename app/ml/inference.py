"""Pemuatan model dan inferensi SpotCheck.

PENTING — preprocessing di modul ini harus sama persis dengan pipeline saat
training (notebook v2-KL-r0-SDS, fungsi transform_image dan
preprocess_for_inference): kanonisasi orientasi, letterbox resize ke kanvas
336x224, lalu normalisasi /255. Model tidak punya layer Rescaling di dalamnya,
jadi pembagian 255 memang harus dilakukan di sini. Bila langkah ini berbeda dari
saat training, prediksi akan meleset tanpa memunculkan error apa pun.

Konvensi kelas (dari training): 0 = dermatitis, 1 = dermatophytosis.
Output model adalah satu nilai sigmoid = P(dermatophytosis).
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Ukuran kanvas model: 336x224 (rasio 3:2), sama dengan IMAGE_SIZE di notebook.
# Urutan tuple mengikuti konvensi PIL (lebar, tinggi); bentuk tensor yang
# dihasilkan preprocess() adalah (1, 224, 336, 3) mengikuti konvensi TensorFlow.
IMAGE_SIZE = (336, 224)

# Ambang keputusan; notebook memakai 0.5 saat evaluasi.
THRESHOLD = 0.5

# Model disimpan di level modul agar hanya dimuat sekali, lalu dipakai ulang
# untuk setiap request.
_model = None

# Path model diingat terpisah agar model bisa dimuat belakangan (lazy), yaitu
# saat request pertama. Lihat catatan tentang fork di load_model().
_model_path = None


def configure(model_path):
    """Ingat lokasi model tanpa memuatnya.

    Dipakai create_app() supaya pemuatan model bisa ditunda sampai request
    pertama pada server yang melakukan fork (lihat load_model()).
    """
    global _model_path
    _model_path = str(model_path) if model_path else None


def load_model(model_path=None):
    """Muat model Keras sekali dan simpan di level modul.

    PENTING soal fork: TensorFlow tidak aman terhadap fork(). Bila TF sudah
    diimpor dan model sudah dimuat di proses induk, lalu server menggandakan
    diri (fork) untuk membuat proses pekerja — seperti yang dilakukan Phusion
    Passenger di cPanel — maka proses anak mewarisi runtime TF tanpa thread
    internalnya. Akibatnya model.predict() MENGGANTUNG tanpa pesan error.

    Karena itu di lingkungan yang mem-fork, model harus dimuat di dalam proses
    pekerja (lewat get_model() saat request pertama), bukan saat startup.
    Setel EAGER_LOAD_MODEL=0 untuk perilaku itu.

    Args:
        model_path: Path ke berkas .keras. Bila None, memakai path yang sudah
            disetel lewat configure().

    Returns:
        Model Keras yang sudah dimuat.

    Raises:
        FileNotFoundError: Bila berkas model tidak ada di path tersebut.
    """
    global _model, _model_path
    if model_path is not None:
        _model_path = str(model_path)

    if _model is None:
        path = Path(_model_path) if _model_path else None
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Berkas model tidak ditemukan: {_model_path!r}. "
                "Periksa MODEL_PATH di .env atau keberadaan "
                "app/ml/model_final_best.keras."
            )

        # Impor TensorFlow ditunda sampai di sini supaya proses impor modul
        # tetap ringan (TF butuh beberapa detik untuk diimpor) dan agar TF tidak
        # ikut terbawa fork bila pemuatannya sengaja ditunda.
        import tensorflow as tf

        _model = tf.keras.models.load_model(path)
    return _model


def get_model():
    """Kembalikan model, memuatnya lebih dulu bila belum dimuat.

    Raises:
        RuntimeError: Bila configure() maupun load_model() belum pernah dipanggil.
    """
    if _model is None:
        if not _model_path:
            raise RuntimeError(
                "Model belum dikonfigurasi. Panggil configure() atau load_model() "
                "lebih dulu."
            )
        return load_model()
    return _model


def canonical_orientation(img):
    """Putar citra potret 90 derajat agar semua citra menjadi lanskap.

    Kanvas model berbentuk lanskap 3:2. Tanpa langkah ini, foto potret akan
    dikecilkan sampai muat pada tinggi kanvas dan menyisakan bantalan hitam yang
    sangat lebar di kiri-kanan, sehingga area kulitnya jauh lebih kecil daripada
    yang dilihat model saat training. Notebook menerapkan pemutaran yang sama
    pada SELURUH data (latih, validasi, uji), jadi langkah ini wajib ada di sini.

    Args:
        img: Citra PIL.

    Returns:
        Citra PIL yang sisi lebarnya >= sisi tingginya.
    """
    width, height = img.size
    if height > width:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


def letterbox_resize(img, target_size, fill_color=(0, 0, 0)):
    """Ubah ukuran citra secara proporsional lalu beri padding di tengah.

    Rasio aspek citra asli dipertahankan (tidak digepengkan); sisa ruang diisi
    warna fill_color. Ini strategi resize yang dipakai saat training.

    Sesuai notebook, sisi hasil dijaga minimal 1 piksel agar citra yang sangat
    panjang dan tipis tidak menghasilkan ukuran nol.

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
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    resized_img = img.resize((new_w, new_h), Image.BILINEAR)

    padded_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))
    return padded_img


def preprocess(image_file):
    """Ubah berkas citra menjadi array siap-prediksi (1, 224, 336, 3).

    Urutan langkahnya mengikuti preprocess_for_inference() di notebook, dengan
    satu tambahan di depan: pelurusan EXIF (lihat komentar di dalam fungsi).

    Args:
        image_file: Path, objek file, atau stream yang bisa dibaca PIL.

    Returns:
        Array float32 ternormalisasi 0..1 dengan batch berukuran 1.

    Raises:
        PIL.UnidentifiedImageError: Bila berkas bukan citra yang valid.
    """
    img = Image.open(image_file)
    # Kamera ponsel tidak memutar piksel; mereka menyimpan orientasi sebagai tag
    # EXIF. Browser menghormati tag itu (preview tampak tegak), Pillow tidak —
    # sehingga tanpa langkah ini model menerima foto MIRING padahal pengguna
    # melihatnya tegak.
    # Pada citra tanpa tag EXIF (seperti dataset training) fungsi ini tidak
    # mengubah apa pun, jadi pipeline-nya tetap sama dengan saat training. Ia
    # harus berjalan SEBELUM canonical_orientation(), supaya keputusan
    # potret/lanskap dibuat atas orientasi yang sebenarnya dilihat pengguna.
    img = ImageOps.exif_transpose(img)              # 0. luruskan sesuai EXIF
    img = img.convert("RGB")                        # 1. paksa RGB
    img = canonical_orientation(img)                # 2. potret -> lanskap
    img = letterbox_resize(img, IMAGE_SIZE)         # 3. letterbox ke 336x224
    arr = np.asarray(img, dtype="float32") / 255.0  # 4. normalisasi 0..1
    arr = np.expand_dims(arr, axis=0)               # 5. batch 1 -> (1,224,336,3)
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

    # Output sigmoid tunggal = P(dermatophytosis), kelas positif saat training.
    p_dermatophytosis = float(model.predict(arr, verbose=0)[0][0])

    is_dermatophytosis = p_dermatophytosis >= THRESHOLD
    return {
        "verdict": "Dermatophytosis" if is_dermatophytosis else "Dermatitis",
        "confidence": round(
            max(p_dermatophytosis, 1 - p_dermatophytosis) * 100
        ),
        "dermatitis_pct": round((1 - p_dermatophytosis) * 100),
        "dermatophytosis_pct": round(p_dermatophytosis * 100),
        # Probabilitas mentah disertakan untuk keperluan debugging/analisis.
        "probability_dermatophytosis": p_dermatophytosis,
    }

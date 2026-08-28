"""Uji preprocessing dan inferensi.

Fokus utama berkas ini: membuktikan bahwa preprocessing aplikasi sama persis
dengan pipeline saat training. Bila uji ini gagal, prediksi aplikasi tidak lagi
bisa dipercaya meskipun aplikasinya tetap berjalan tanpa error.

Acuan: notebook v2-kl-r0-sds-dd-classification-44, bagian 4.3 (transformasi
citra) dan 7.1 (pemrosesan inferensi).
"""

import json

import numpy as np
import pytest
from PIL import Image

from app.ml import inference
from tests.conftest import make_jpeg

# Kanvas model: 336 x 224 (lebar x tinggi, konvensi PIL).
LEBAR, TINGGI = inference.IMAGE_SIZE


# --------------------------------------------------------------------------
# Kesesuaian dengan berkas konfigurasi hasil ekspor notebook
# --------------------------------------------------------------------------

def test_constants_match_exported_notebook_config():
    """Konstanta modul harus sama dengan yang diekspor notebook.

    app/ml/inference_config.json ditulis oleh notebook saat mengekspor model,
    jadi berkas itulah sumber kebenarannya. Uji ini menangkap kasus model
    diganti tanpa preprocessing ikut disesuaikan — kegagalan yang tidak
    memunculkan error apa pun saat aplikasi berjalan.
    """
    from tests.conftest import PROJECT_ROOT

    config_path = PROJECT_ROOT / "app" / "ml" / "inference_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert inference.IMAGE_SIZE == (config["canvas_width"], config["canvas_height"])
    assert inference.THRESHOLD == config["threshold"]
    assert config["class_names"] == ["dermatitis", "dermatophytosis"]
    assert config["positive_class"] == "dermatophytosis"


def test_model_input_shape_matches_preprocess_output(app, eczema_photo):
    """Bentuk keluaran preprocess() harus diterima model tanpa penyesuaian."""
    model_shape = inference.get_model().input_shape[1:]
    assert inference.preprocess(eczema_photo).shape[1:] == model_shape


# --------------------------------------------------------------------------
# Kanonisasi orientasi: potret diputar menjadi lanskap
# --------------------------------------------------------------------------

def test_canonical_orientation_rotates_portrait():
    """Citra tegak harus menjadi mendatar, tanpa piksel yang hilang."""
    img = Image.new("RGB", (300, 900), (255, 0, 0))
    assert inference.canonical_orientation(img).size == (900, 300)


def test_canonical_orientation_leaves_landscape_alone():
    img = Image.new("RGB", (900, 300), (255, 0, 0))
    assert inference.canonical_orientation(img) is img


def test_canonical_orientation_leaves_square_alone():
    """Sisi sama panjang tidak punya orientasi untuk dikanonisasi."""
    img = Image.new("RGB", (500, 500), (255, 0, 0))
    assert inference.canonical_orientation(img) is img


def test_canonical_orientation_keeps_every_pixel():
    """Pemutaran 90 derajat memindahkan piksel, bukan membuangnya."""
    rng = np.random.default_rng(0)
    asli = Image.fromarray(rng.integers(0, 256, (900, 300, 3), dtype=np.uint8))
    diputar = inference.canonical_orientation(asli)

    assert sorted(np.asarray(asli).ravel()) == sorted(np.asarray(diputar).ravel())


def test_portrait_photo_fills_more_of_the_canvas_after_rotation():
    """Inilah alasan kanonisasi ada: foto tegak tidak lagi jadi strip tipis.

    Tanpa pemutaran, foto 300x900 hanya mengisi 75 dari 336 kolom kanvas.
    """
    img = Image.new("RGB", (300, 900), (255, 0, 0))

    tanpa_putar = np.asarray(inference.letterbox_resize(img, inference.IMAGE_SIZE))
    dengan_putar = np.asarray(
        inference.letterbox_resize(
            inference.canonical_orientation(img), inference.IMAGE_SIZE
        )
    )

    kolom_tanpa = sum(tanpa_putar[:, x].max() > 0 for x in range(LEBAR))
    kolom_dengan = sum(dengan_putar[:, x].max() > 0 for x in range(LEBAR))

    assert kolom_dengan > kolom_tanpa * 3


# --------------------------------------------------------------------------
# Letterbox: rasio aspek dipertahankan, padding hitam, citra di tengah
# --------------------------------------------------------------------------

def test_letterbox_returns_target_size():
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    assert inference.letterbox_resize(img, inference.IMAGE_SIZE).size == (LEBAR, TINGGI)


def test_letterbox_preserves_aspect_ratio():
    """Citra 1000x200 (rasio 5:1) harus tetap 5:1 setelah diskalakan."""
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    arr = np.asarray(inference.letterbox_resize(img, inference.IMAGE_SIZE))

    rows = [y for y in range(TINGGI) if arr[y].max() > 0]
    cols = [x for x in range(LEBAR) if arr[:, x].max() > 0]

    # Skala = min(336/1000, 224/200) = 0.336 -> tinggi 200*0.336 = 67.2 -> int() = 67
    assert len(rows) == 67
    assert len(cols) == LEBAR


def test_letterbox_centers_the_image():
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    arr = np.asarray(inference.letterbox_resize(img, inference.IMAGE_SIZE))

    rows = [y for y in range(TINGGI) if arr[y].max() > 0]
    padding_atas = rows[0]
    padding_bawah = TINGGI - 1 - rows[-1]
    assert abs(padding_atas - padding_bawah) <= 1


def test_letterbox_pads_with_black():
    img = Image.new("RGB", (1000, 200), (255, 0, 0))
    arr = np.asarray(inference.letterbox_resize(img, inference.IMAGE_SIZE))
    assert arr[0].max() == 0             # baris paling atas = padding
    assert arr[TINGGI - 1].max() == 0    # baris paling bawah = padding


def test_letterbox_matching_ratio_needs_no_padding():
    """Citra 3:2 sama dengan rasio kanvas, jadi mengisi penuh tanpa bantalan."""
    img = Image.new("RGB", (900, 600), (0, 0, 255))
    arr = np.asarray(inference.letterbox_resize(img, inference.IMAGE_SIZE))
    assert all(arr[y].max() > 0 for y in range(TINGGI))
    assert all(arr[:, x].max() > 0 for x in range(LEBAR))


def test_letterbox_square_image_is_padded_sideways():
    """Pada kanvas 3:2, citra persegi mengisi tinggi penuh dan dibantali kiri-kanan."""
    img = Image.new("RGB", (500, 500), (0, 0, 255))
    arr = np.asarray(inference.letterbox_resize(img, inference.IMAGE_SIZE))

    assert all(arr[y].max() > 0 for y in range(TINGGI))   # tinggi terisi penuh
    assert arr[:, 0].max() == 0                           # kolom terkiri = padding
    assert arr[:, LEBAR - 1].max() == 0                   # kolom terkanan = padding


def test_letterbox_never_collapses_a_thin_image():
    """Citra sangat panjang dan tipis tidak boleh menghasilkan sisi nol piksel."""
    img = Image.new("RGB", (2000, 7), (255, 0, 0))
    arr = np.asarray(inference.letterbox_resize(img, inference.IMAGE_SIZE))
    assert arr.max() > 0, "citra menyusut sampai hilang"


# --------------------------------------------------------------------------
# preprocess(): bentuk, tipe, rentang nilai
# --------------------------------------------------------------------------

def test_preprocess_shape_and_dtype(eczema_photo):
    arr = inference.preprocess(eczema_photo)
    assert arr.shape == (1, TINGGI, LEBAR, 3)
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
    assert inference.preprocess(buf).shape == (1, TINGGI, LEBAR, 3)


def test_preprocess_rejects_non_image():
    import io

    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        inference.preprocess(io.BytesIO(b"ini jelas bukan citra"))


@pytest.mark.parametrize("size", [(1000, 200), (200, 1000), (500, 500), (50, 37)])
def test_preprocess_handles_various_shapes(size):
    assert inference.preprocess(make_jpeg(size)).shape == (1, TINGGI, LEBAR, 3)


def test_preprocess_accepts_webp():
    import io

    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (190, 120, 110)).save(buf, "WEBP")
    buf.seek(0)
    assert inference.preprocess(buf).shape == (1, TINGGI, LEBAR, 3)


# --------------------------------------------------------------------------
# Orientasi EXIF
# --------------------------------------------------------------------------

def _foto_ponsel(img, orientation=6):
    """Tiru foto ponsel: piksel tersimpan miring + tag EXIF orientasi."""
    import io

    exif = Image.Exif()
    exif[274] = orientation  # 274 = tag Orientation; 6 = "putar 90 derajat saat ditampilkan"
    buf = io.BytesIO()
    img.rotate(90, expand=True).save(buf, "JPEG", exif=exif, quality=95)
    buf.seek(0)
    return buf


def test_preprocess_applies_exif_orientation(tinea_photo):
    """Foto ber-EXIF harus diluruskan agar sama dengan versi tegaknya.

    Kamera ponsel tidak memutar piksel, hanya menandai orientasi lewat EXIF.
    Browser menghormati tag itu (preview tampak tegak) sedangkan Pillow tidak,
    sehingga tanpa exif_transpose model akan menerima foto miring padahal
    pengguna melihatnya tegak.
    """
    tegak = inference.preprocess(tinea_photo)
    dari_ponsel = inference.preprocess(_foto_ponsel(Image.open(tinea_photo).convert("RGB")))

    # Tidak bisa identik bit-per-bit (ada siklus JPEG ulang), tapi harus sangat mirip.
    selisih = np.abs(tegak - dari_ponsel).mean()
    assert selisih < 0.02, f"foto ber-EXIF tidak diluruskan (selisih rata-rata {selisih:.4f})"


def test_canonical_orientation_alone_cannot_replace_exif_transpose(tinea_photo):
    """Kanonisasi orientasi tidak membuat exif_transpose jadi mubazir.

    Keduanya sama-sama memutar citra, jadi wajar bila muncul dugaan salah
    satunya berlebihan. Uji ini membuktikan tidak: kanonisasi hanya menjamin
    citra menjadi mendatar, sedangkan MENDATAR YANG MANA ditentukan oleh tag
    EXIF. Tanpa exif_transpose, foto ponsel berakhir terbalik 180 derajat.
    """
    dengan_exif = inference.preprocess(
        _foto_ponsel(Image.open(tinea_photo).convert("RGB"))
    )

    # Ulangi pipeline tanpa langkah exif_transpose.
    from tests.conftest import preprocess_tanpa_exif

    tanpa_exif = preprocess_tanpa_exif(
        _foto_ponsel(Image.open(tinea_photo).convert("RGB"))
    )

    selisih = np.abs(dengan_exif - tanpa_exif).mean()
    assert selisih > 0.02, (
        "melewatkan exif_transpose ternyata tidak mengubah masukan model; "
        f"selisih rata-rata hanya {selisih:.4f}"
    )


# --------------------------------------------------------------------------
# Uji paling penting: preprocessing identik dengan notebook
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sumber",
    ["eczema", "tinea", (1000, 200), (200, 1000), (500, 500), (50, 37)],
)
def test_preprocess_matches_training_pipeline(sumber, eczema_photo, tinea_photo):
    """preprocess() harus sama persis dengan pipeline notebook.

    Notebook v2-KL-r0-SDS, preprocess_for_inference():
        img = Image.open(path).convert("RGB")
        img = transform_image(img)            # kanonisasi orientasi + letterbox
        arr = np.asarray(img, np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

    Fungsi acuan di bawah disalin verbatim dari notebook, bukan memanggil ulang
    fungsi aplikasi — kalau tidak, uji ini hanya akan membandingkan kode dengan
    dirinya sendiri.
    """
    IMAGE_SIZE = (336, 224)

    def canonical_orientation(img):
        width, height = img.size
        if height > width:
            return img.transpose(Image.Transpose.ROTATE_90)
        return img

    def letterbox_resize(img, target_size=IMAGE_SIZE, fill_color=(0, 0, 0)):
        target_width, target_height = target_size
        original_width, original_height = img.size
        scale = min(target_width / original_width, target_height / original_height)
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))
        resized = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", target_size, fill_color)
        canvas.paste(resized, ((target_width - new_width) // 2,
                               (target_height - new_height) // 2))
        return canvas

    def transform_image(img, target_size=IMAGE_SIZE):
        return letterbox_resize(canonical_orientation(img), target_size)

    if sumber == "eczema":
        masukan, ulangan = eczema_photo, eczema_photo
    elif sumber == "tinea":
        masukan, ulangan = tinea_photo, tinea_photo
    else:
        masukan, ulangan = make_jpeg(sumber), make_jpeg(sumber)

    with Image.open(ulangan) as img:
        transformed = transform_image(img.convert("RGB"))
    referensi = np.expand_dims(
        np.asarray(transformed, dtype=np.float32) / 255.0, axis=0
    )

    hasil = inference.preprocess(masukan)

    assert hasil.shape == referensi.shape
    assert hasil.dtype == referensi.dtype
    assert np.array_equal(hasil, referensi), "preprocessing menyimpang dari pipeline training"


# --------------------------------------------------------------------------
# predict(): kontrak keluaran
# --------------------------------------------------------------------------

def test_predict_returns_expected_keys(app, eczema_photo):
    hasil = inference.predict(eczema_photo)
    assert set(hasil) == {
        "verdict", "confidence", "dermatitis_pct", "dermatophytosis_pct",
        "probability_dermatophytosis",
    }


def test_predict_percentages_sum_to_100(app, eczema_photo):
    hasil = inference.predict(eczema_photo)
    assert hasil["dermatitis_pct"] + hasil["dermatophytosis_pct"] == 100


def test_predict_confidence_is_the_larger_class(app, tinea_photo):
    hasil = inference.predict(tinea_photo)
    assert hasil["confidence"] == max(
        hasil["dermatitis_pct"], hasil["dermatophytosis_pct"]
    )
    assert 50 <= hasil["confidence"] <= 100


def test_predict_verdict_follows_threshold(app, eczema_photo, tinea_photo):
    """Verdict harus konsisten dengan ambang 0.5 pada P(dermatophytosis)."""
    for path in (eczema_photo, tinea_photo):
        hasil = inference.predict(path)
        jamur_menang = hasil["probability_dermatophytosis"] >= inference.THRESHOLD
        assert (hasil["verdict"] == "Dermatophytosis") == jamur_menang


def test_predict_probability_within_unit_range(app, eczema_photo):
    assert 0.0 <= inference.predict(eczema_photo)["probability_dermatophytosis"] <= 1.0


def test_predict_is_deterministic(app, tinea_photo):
    """Citra yang sama harus menghasilkan probabilitas yang sama persis."""
    a = inference.predict(tinea_photo)["probability_dermatophytosis"]
    b = inference.predict(tinea_photo)["probability_dermatophytosis"]
    assert a == b


def test_predict_classifies_the_ringworm_photo(app, tinea_photo):
    """Uji kewarasan menyeluruh: preprocessing + model + pemetaan kelas.

    Foto ini adalah ringworm, anggota kelompok dermatophytosis. Bila pemetaan
    0=dermatitis / 1=dermatophytosis tertukar, uji ini akan gagal.
    """
    assert inference.predict(tinea_photo)["verdict"] == "Dermatophytosis"


def test_predict_classifies_the_eczema_photo(app, eczema_photo):
    """Sisi dermatitis dari uji kewarasan di atas.

    Kedua sisi perlu diuji: bila hanya satu kelas yang diperiksa, model yang
    selalu menjawab kelas itu tetap lolos. Foto eczema versi sebelumnya tidak
    bisa dipakai di sini karena salah dinilai sebagai dermatophytosis; foto
    pengganti pada halaman edukasi dinilai dengan benar.
    """
    assert inference.predict(eczema_photo)["verdict"] == "Dermatitis"


# --------------------------------------------------------------------------
# Pemuatan model
# --------------------------------------------------------------------------

def test_load_model_with_missing_file_raises_clear_error(tmp_path):
    """Path model yang salah harus memberi FileNotFoundError yang jelas."""
    # Model dan path-nya disimpan di level modul dan dipakai bersama seluruh
    # sesi uji, jadi keduanya wajib dikembalikan agar uji lain tidak terganggu.
    model_asli, path_asli = inference._model, inference._model_path
    inference._model = None
    try:
        with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
            inference.load_model(tmp_path / "tidak_ada.keras")
        with pytest.raises(FileNotFoundError):
            inference.load_model("")   # meniru MODEL_PATH kosong di .env
    finally:
        inference._model, inference._model_path = model_asli, path_asli


def test_get_model_without_configure_raises():
    """Tanpa configure() maupun load_model(), get_model() harus menolak."""
    model_asli, path_asli = inference._model, inference._model_path
    inference._model = None
    inference._model_path = None
    try:
        with pytest.raises(RuntimeError):
            inference.get_model()
    finally:
        inference._model, inference._model_path = model_asli, path_asli


def test_get_model_loads_lazily_when_only_configured():
    """Bila baru dikonfigurasi, get_model() memuat model sendiri (lazy).

    Perilaku ini yang dipakai di Passenger/cPanel: model sengaja tidak dimuat
    saat startup agar tidak terbawa fork, lalu dimuat saat request pertama.
    """
    from tests.conftest import PROJECT_ROOT

    model_asli, path_asli = inference._model, inference._model_path
    inference._model = None
    inference.configure(PROJECT_ROOT / "app" / "ml" / "model_final_best.keras")
    try:
        assert inference.get_model() is not None
    finally:
        inference._model, inference._model_path = model_asli, path_asli

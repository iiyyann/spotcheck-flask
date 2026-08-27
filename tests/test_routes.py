"""Uji route HTTP: halaman utama dan endpoint /predict."""

import io

import pytest

from tests.conftest import make_jpeg


# --------------------------------------------------------------------------
# Halaman utama
# --------------------------------------------------------------------------

def test_index_returns_ok(client):
    assert client.get("/").status_code == 200


def test_index_has_all_four_pages(client):
    html = client.get("/").get_data(as_text=True)
    for page_id in ("page-home", "page-dermatitis", "page-dermatophytosis",
                    "page-about"):
        assert f'id="{page_id}"' in html


def test_index_has_no_prototype_leftovers(client):
    """Banner prototype dan kata "Streamlit" tidak boleh muncul (CLAUDE.md §2)."""
    html = client.get("/").get_data(as_text=True)
    assert "proto-note" not in html
    assert "Streamlit" not in html
    assert "runDemo" not in html


def test_index_keeps_the_medical_disclaimer(client):
    html = client.get("/").get_data(as_text=True)
    assert "This is not a medical diagnosis." in html


def test_index_links_the_favicon(client):
    html = client.get("/").get_data(as_text=True)
    assert 'rel="icon"' in html
    assert "img/favicon.svg" in html
    assert "apple-touch-icon" in html


@pytest.mark.parametrize(
    "path",
    [
        "/static/img/favicon.svg",
        "/static/img/favicon-32.png",
        "/static/img/favicon-16.png",
        "/static/img/apple-touch-icon.png",
    ],
)
def test_favicon_files_are_served(client, path):
    res = client.get(path)
    assert res.status_code == 200
    assert res.data, "berkas ikon kosong"


def test_favicon_ico_redirects(client):
    """Perkakas dan crawler tetap meminta /favicon.ico di root."""
    res = client.get("/favicon.ico")
    assert res.status_code in (301, 302, 308)
    assert "favicon.svg" in res.headers["Location"]


@pytest.mark.parametrize(
    "path",
    [
        "/static/img/model/training-curves.png",
        "/static/img/model/roc-curve.png",
        "/static/img/model/inference-dermatitis.png",
        "/static/img/model/inference-dermatophytosis.png",
    ],
)
def test_model_figures_are_served(client, path):
    res = client.get(path)
    assert res.status_code == 200
    assert res.data


@pytest.mark.parametrize(
    "nama, lebar_min, lebar_tampil",
    [
        ("training-curves.png", 1100, 830),
        ("roc-curve.png", 580, 560),
        ("inference-dermatitis.png", 1100, 830),
        ("inference-dermatophytosis.png", 1100, 830),
    ],
)
def test_model_figures_keep_native_resolution(nama, lebar_min, lebar_tampil):
    """Gambar harus berasal dari notebook, bukan hasil salin dari layar.

    Menyalin gambar dari penampil notebook menghasilkan versi selebar viewport
    (529 px), yang lalu diregangkan di halaman dan terlihat blur. Gambar asli
    tertanam di dalam .ipynb dan jauh lebih besar; uji ini menahan agar versi
    kecil tidak masuk kembali.
    """
    from PIL import Image

    from tests.conftest import PROJECT_ROOT

    berkas = PROJECT_ROOT / "app" / "static" / "img" / "model" / nama
    lebar = Image.open(berkas).size[0]

    assert lebar >= lebar_min, f"{nama} hanya {lebar} px — kemungkinan hasil salin layar"
    assert lebar >= lebar_tampil, (
        f"{nama} ({lebar} px) lebih sempit dari lebar tampilnya ({lebar_tampil} px) "
        "sehingga akan diregangkan dan terlihat blur"
    )


def test_about_shows_model_figures_and_metrics(client):
    """Angka pelatihan berasal dari notebook — jangan sampai berubah diam-diam.

    Sumber: v2-kl-r0-sds-dd-classification-44, bagian 6.1 (kinerja per subset),
    6.2 (confusion matrix), 6.3 (metrik turunan) dan 6.4 (ROC-AUC).
    """
    html = client.get("/").get_data(as_text=True)

    for gambar in ("training-curves.png", "roc-curve.png",
                   "inference-dermatitis.png", "inference-dermatophytosis.png"):
        assert gambar in html

    # 6.1 — Train / Validation / Test: loss, accuracy, ROC-AUC.
    for angka in ("0.3065", "89.83%", "96.87%",
                  "0.4297", "83.41%", "91.13%",
                  "0.4116", "86.03%", "91.80%"):
        assert angka in html, f"metrik {angka} hilang dari halaman About"

    # 6.2 — komponen confusion matrix pada 458 citra uji.
    for sel in (">228<", ">27<", ">37<", ">166<"):
        assert sel in html, f"sel confusion matrix {sel} hilang dari halaman About"

    # 6.3 — recall per kelas, angka yang paling sering dikutip di Bab 3.
    for angka in ("89.41%", "81.77%", "85.77%"):
        assert angka in html, f"metrik {angka} hilang dari halaman About"

    # 6.4 — AUC eksak dari kurva ROC.
    assert "0.9176" in html


def test_about_reports_the_dataset_used_for_training(client):
    """Jumlah citra dan pembagiannya harus sesuai notebook (bagian 4.4.1)."""
    html = client.get("/").get_data(as_text=True)
    for angka in ("9,987", "3,049", "2,133", "1,696", "1,353"):
        assert angka in html, f"angka dataset {angka} hilang dari halaman About"


def test_footer_links_to_cleveland_clinic(client):
    html = client.get("/").get_data(as_text=True)
    assert 'href="https://my.clevelandclinic.org/"' in html


def test_eczema_types_have_definitions(client):
    """Tiap tipe eczema harus punya penjelasan, bukan sekadar daftar nama.

    Eczema adalah contoh yang dibahas mendalam pada halaman Dermatitis; daftar
    tipe ini bagian dari materi tersebut.
    """
    html = client.get("/").get_data(as_text=True)
    for tipe in (
        "Atopic dermatitis",
        "Contact dermatitis",
        "Dyshidrotic eczema",
        "Neurodermatitis",
        "Nummular eczema",
        "Seborrheic dermatitis",
    ):
        assert f"<dt>{tipe}</dt>" in html
    assert html.count("<dd>") >= 6


def test_education_pages_state_their_scope(client):
    """Halaman edukasi wajib menyatakan bahwa isinya satu contoh dari kelompok.

    Model mengklasifikasikan KELOMPOK penyakit (dermatitis dan dermatophytosis),
    sedangkan materi edukasinya membahas satu anggota paling dikenal dari tiap
    kelompok: eczema dan ringworm. Tanpa catatan ini pembaca bisa mengira model
    memprediksi eczema atau ringworm secara spesifik.
    """
    html = client.get("/").get_data(as_text=True)

    assert html.count('class="scope-note"') >= 3, (
        "catatan cakupan hilang dari beranda dan/atau halaman edukasi"
    )
    assert "best-known" in html


def test_index_warns_that_out_of_scope_photos_still_get_an_answer(client):
    """Batas terpenting model harus dinyatakan, bukan disimpulkan sendiri.

    Model adalah pengklasifikasi biner closed-set: tidak ada kelas penolakan
    dan tidak ada deteksi out-of-distribution, sehingga citra apa pun tetap
    dijawab salah satu dari dua kelas — kadang dengan keyakinan tinggi.
    Peringatan ini menyangkut keselamatan pengguna, jadi dikunci di sini.
    """
    html = client.get("/").get_data(as_text=True)

    assert 'neither' in html, "peringatan citra di luar kelas hilang dari halaman"
    assert "closed-set" in html, "penjelasan teknis closed-set hilang dari About"
    assert "even when the photo is neither" in html, (
        "disclaimer hasil tidak lagi menyebut bahwa model selalu memilih satu kelas"
    )


def test_index_serves_photos_as_static_files(client):
    html = client.get("/").get_data(as_text=True)
    assert "base64," not in html
    assert "/static/img/eczema.jpg" in html
    assert "/static/img/tinea.jpg" in html


# --------------------------------------------------------------------------
# /predict — jalur normal
# --------------------------------------------------------------------------

def test_predict_returns_result_for_valid_photo(client, eczema_photo):
    with open(eczema_photo, "rb") as f:
        res = client.post(
            "/predict",
            data={"image": (io.BytesIO(f.read()), "eczema.jpg")},
            content_type="multipart/form-data",
        )

    assert res.status_code == 200
    data = res.get_json()
    assert data["verdict"] in ("Dermatitis", "Dermatophytosis")
    assert data["dermatitis_pct"] + data["dermatophytosis_pct"] == 100
    assert 50 <= data["confidence"] <= 100


def test_predict_accepts_png(client):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (180, 120, 110)).save(buf, "PNG")
    buf.seek(0)

    res = client.post(
        "/predict",
        data={"image": (buf, "sampel.png")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200


def test_predict_accepts_uppercase_extension(client):
    res = client.post(
        "/predict",
        data={"image": (make_jpeg((300, 300)), "FOTO.JPG")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200


# --------------------------------------------------------------------------
# /predict — validasi
# --------------------------------------------------------------------------

def test_predict_without_file_returns_400(client):
    res = client.post("/predict", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_with_empty_filename_returns_400(client):
    res = client.post(
        "/predict",
        data={"image": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_rejects_disallowed_extension(client):
    res = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"halo"), "catatan.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_rejects_file_without_extension(client):
    res = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"halo"), "berkas")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_predict_rejects_fake_image(client):
    """Berkas .jpg yang isinya bukan citra harus ditolak dengan rapi, bukan crash."""
    res = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"ini hanya teks biasa"), "palsu.jpg")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_rejects_oversized_upload(client, app):
    """Upload melebihi MAX_CONTENT_LENGTH harus jadi 413 berformat JSON."""
    batas = app.config["MAX_CONTENT_LENGTH"]
    besar = b"\xff\xd8" + b"0" * (batas + 1024)

    res = client.post(
        "/predict",
        data={"image": (io.BytesIO(besar), "besar.jpg")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 413
    assert "error" in res.get_json()


def test_predict_errors_are_json_not_html(client):
    """Klien selalu mem-parse JSON, jadi error pun harus JSON."""
    res = client.post("/predict", data={}, content_type="multipart/form-data")
    assert res.is_json


def test_predict_rejects_get_method(client):
    assert client.get("/predict").status_code == 405

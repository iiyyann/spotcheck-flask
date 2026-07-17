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
    for page_id in ("page-home", "page-eczema", "page-tinea", "page-about"):
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
        "/static/img/model/inference-samples.png",
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
        ("inference-samples.png", 1100, 830),
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
    """Angka pelatihan berasal dari notebook — jangan sampai berubah diam-diam."""
    html = client.get("/").get_data(as_text=True)

    for gambar in ("training-curves.png", "roc-curve.png", "inference-samples.png"):
        assert gambar in html

    # Train / Validation / Test — loss dan accuracy, persis seperti notebook.
    for angka in ("0.4463", "83.80%", "0.4401", "83.83%", "0.4721", "83.17%"):
        assert angka in html, f"metrik {angka} hilang dari halaman About"


def test_footer_links_to_cleveland_clinic(client):
    html = client.get("/").get_data(as_text=True)
    assert 'href="https://my.clevelandclinic.org/"' in html


def test_eczema_types_have_definitions(client):
    """Tiap tipe eczema harus punya penjelasan, bukan sekadar daftar nama."""
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
    assert data["verdict"] in ("Eczema", "Tinea")
    assert data["eczema_pct"] + data["tinea_pct"] == 100
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

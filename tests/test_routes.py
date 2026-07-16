"""Uji route HTTP: halaman utama dan endpoint /predict."""

import io

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

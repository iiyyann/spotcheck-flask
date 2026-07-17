"""Route HTTP SpotCheck.

Route dijaga tetap tipis: hanya menangani request/response dan validasi upload.
Seluruh logika model berada di app/ml/inference.py.
"""

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from PIL import UnidentifiedImageError

from app.ml import inference

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    """Sajikan halaman tunggal (navigasi antar-halaman dilakukan di sisi klien)."""
    return render_template("index.html")


@bp.get("/favicon.ico")
def favicon():
    """Arahkan /favicon.ico ke ikon SVG.

    Browser memakai tag <link rel="icon"> di base.html, tetapi sebagian perkakas
    dan crawler tetap meminta /favicon.ico di root. Tanpa route ini, permintaan
    itu memenuhi log dengan 404.
    """
    return redirect(url_for("static", filename="img/favicon.svg"))


@bp.post("/predict")
def predict():
    """Terima satu citra kulit, kembalikan hasil klasifikasi sebagai JSON.

    Request: multipart/form-data dengan field "image".
    Response 200: {verdict, confidence, eczema_pct, tinea_pct, probability_tinea}
    Response 400: {error} bila berkas tidak ada / tipe salah / bukan citra valid.
    """
    if "image" not in request.files:
        return jsonify(error="Tidak ada berkas yang diunggah."), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify(error="Tidak ada berkas yang dipilih."), 400

    if not _is_allowed(file.filename):
        allowed = ", ".join(sorted(current_app.config["ALLOWED_EXTENSIONS"])).upper()
        return jsonify(error=f"Format tidak didukung. Gunakan {allowed}."), 400

    try:
        result = inference.predict(file.stream)
    except UnidentifiedImageError:
        return jsonify(error="Berkas ini bukan citra yang valid."), 400
    except Exception:  # noqa: BLE001 - jangan bocorkan detail internal ke klien
        current_app.logger.exception("Prediksi gagal")
        return jsonify(error="Gagal menganalisis citra. Coba lagi."), 500

    return jsonify(result)


@bp.errorhandler(413)
def handle_too_large(_error):
    """Ubah error bawaan Flask untuk upload kebesaran menjadi JSON yang jelas."""
    limit_mb = current_app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
    return jsonify(error=f"Berkas terlalu besar. Maksimal {limit_mb:.0f} MB."), 413


def _is_allowed(filename):
    """Cek ekstensi berkas terhadap daftar yang diizinkan di config."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in current_app.config["ALLOWED_EXTENSIONS"]

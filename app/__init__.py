"""Application factory SpotCheck.

Tidak ada objek `app` global di level modul: aplikasi selalu dibuat lewat
create_app() agar konfigurasi bisa ditukar (development / production) dan
mudah diuji.
"""

import logging

from flask import Flask

from config import get_config


def create_app(config_name=None):
    """Buat dan konfigurasi instance Flask.

    Args:
        config_name: "development" atau "production". Bila None, dibaca dari
            environment variable FLASK_ENV.

    Returns:
        Instance Flask yang siap dijalankan.
    """
    app = Flask(__name__)
    app.config.from_object(get_config(config_name))

    from app.routes import bp as main_bp

    app.register_blueprint(main_bp)

    _warn_on_default_secret(app)
    _setup_model(app)

    return app


def _setup_model(app):
    """Beritahu lokasi model, dan muat sekarang bila memang diinginkan."""
    from app.ml import inference

    inference.configure(app.config["MODEL_PATH"])

    if app.config["EAGER_LOAD_MODEL"]:
        _load_model(app)


def _warn_on_default_secret(app):
    """Ingatkan bila aplikasi berjalan di produksi dengan SECRET_KEY default."""
    from config import DEFAULT_SECRET_KEY

    if not app.debug and app.config["SECRET_KEY"] == DEFAULT_SECRET_KEY:
        app.logger.warning(
            "SECRET_KEY masih memakai nilai default. Set environment variable "
            "SECRET_KEY sebelum melayani trafik publik."
        )


def _load_model(app):
    """Muat model sekali saat startup agar request pertama tidak menunggu.

    Reloader Flask menjalankan proses dua kali saat debug; pemuatan dilewati di
    proses induk supaya model tidak dimuat sia-sia.
    """
    import os

    if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    from app.ml import inference

    logging.getLogger(__name__).info("Memuat model dari %s", app.config["MODEL_PATH"])
    inference.load_model()

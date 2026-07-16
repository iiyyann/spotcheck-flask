"""Entry point untuk menjalankan SpotCheck di mesin lokal.

Produksi memakai gunicorn dan tidak lewat file ini, contoh:
    gunicorn "app:create_app()" --bind 0.0.0.0:7860
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

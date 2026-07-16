# Image produksi SpotCheck.
#
# Disiapkan untuk Hugging Face Spaces (Docker Space), yang:
#   - mengarahkan trafik ke port 7860
#   - menjalankan container sebagai user non-root ber-UID 1000
# Konfigurasi yang sama juga jalan di Render atau host Docker mana pun; yang
# perlu disesuaikan hanya port bila platformnya berbeda.

FROM python:3.12-slim

# libgomp1 = runtime OpenMP yang ditautkan TensorFlow. Image -slim tidak
# menyertakannya, dan tanpa ini `import tensorflow` gagal dengan
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces menjalankan container sebagai UID 1000.
RUN useradd --create-home --uid 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    FLASK_ENV=production \
    # Port default = 7860 (Hugging Face Spaces). Platform seperti Render
    # menyuntikkan PORT sendiri dan akan menimpa nilai ini.
    PORT=7860 \
    # Satu worker ~300 MB (TensorFlow + model, diukur langsung). Default 1 agar
    # muat di host 512 MB. Naikkan lewat env di host yang memorinya lebih lega.
    WEB_CONCURRENCY=1

WORKDIR $HOME/app

# Dependensi disalin lebih dulu agar layer pip ter-cache dan tidak dibangun
# ulang setiap kali kode aplikasi berubah.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

EXPOSE 7860

# Model dimuat sekali per worker di dalam create_app(). --preload sengaja TIDAK
# dipakai: memuat TensorFlow sebelum fork bisa membuat worker menggantung.
#
# Bentuk shell dipakai agar ${PORT} diekspansi; `exec` membuat gunicorn
# menggantikan proses shell sehingga sinyal SIGTERM dari platform diterima
# gunicorn secara langsung (shutdown tetap rapi). Jumlah worker diambil
# gunicorn dari env WEB_CONCURRENCY.
CMD exec gunicorn "app:create_app()" \
    --bind "0.0.0.0:${PORT}" \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -

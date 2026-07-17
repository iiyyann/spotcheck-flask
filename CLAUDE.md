# CLAUDE.md — SpotCheck (Flask)

> This file gives Claude Code the fixed context for this project. Read it fully
> before writing any code. It defines the goal, the tech stack, the folder
> structure, the coding conventions, and — most importantly — the exact model
> behavior and image preprocessing that must be reproduced. Do not deviate from
> the preprocessing: it must match how the model was trained, or predictions
> will be wrong.

---

## 1. Project overview

SpotCheck is the web deliverable of an undergraduate final project (skripsi /
Proyek Ilmiah). The machine-learning model is **already finished** and exported;
this repository only builds the **web application** around it and prepares it for
**deployment** so other people can use it (not just on the developer's machine).

The app lets a user upload one skin photo and get a supportive read on whether it
looks more like **eczema** or **tinea**, plus educational pages about each
condition. It is explicitly **not** a medical diagnosis tool.

**Working directory (developer's machine, Windows):**
`C:\Users\Reihan\Proyek PI\SpotCheck-Flask`

---

## 2. Tech stack (fixed decisions)

- **Backend:** Flask (Python). Use the **application factory pattern**
  (`create_app()`) and keep configuration in a separate `config.py`.
- **Model runtime:** TensorFlow / Keras. Use `tensorflow-cpu` (not full
  `tensorflow`) to keep the deployment image small — no GPU is needed for
  inference.
- **Server (production):** `gunicorn`.
- **Frontend:** the existing prototype's HTML/CSS/JS, converted to Jinja2
  templates. No frontend build step, no React, no CSS framework — the prototype
  already contains all styling inline.

> ⚠️ The prototype text (`spotcheck-prototype.html`) mentions **"Streamlit"** in a
> few places because it was originally drafted for Streamlit. This project uses
> **Flask**. Everywhere the visible copy says "Streamlit", replace it with neutral
> wording such as "web app" / "the SpotCheck app". Also **remove** the yellow
> `DESIGN PROTOTYPE …` banner (`.proto-note`) at the top of the body — it must not
> appear in the real app.

---

## 3. The model — how to use it correctly

- **File:** `model_final_best.keras` (place it under `app/ml/`).
- **Input shape:** `(224, 224, 3)`, RGB, pixel values scaled to `0–1`.
- **Output:** a single **sigmoid** value in `[0, 1]` = **P(tinea)**.
- **Class mapping (from training):** `0 = eczema`, `1 = tinea`.
- **Decision threshold:** `p >= 0.5` → **Tinea**; `p < 0.5` → **Eczema**.
- **Confidence to display:** `max(p, 1 - p) * 100`, rounded.
- **Bar percentages to display:**
  - Tinea % = `round(p * 100)`
  - Eczema % = `round((1 - p) * 100)`

**Load the model once at application startup** (e.g. in the factory or an
`inference.py` module), never per request.

### Preprocessing — must match training EXACTLY

The model was trained on **letterbox-resized** images (proportional resize +
centered black padding), then normalized by dividing by 255. Reproduce this
precisely for every uploaded image. Reference implementation from the training
notebook:

```python
from PIL import Image
import numpy as np

IMAGE_SIZE = (224, 224)

def letterbox_resize(img, target_size, fill_color=(0, 0, 0)):
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
    img = Image.open(image_file).convert("RGB")     # 1. force RGB
    img = letterbox_resize(img, IMAGE_SIZE)         # 2. letterbox to 224x224
    arr = np.asarray(img, dtype="float32") / 255.0  # 3. normalize 0..1
    arr = np.expand_dims(arr, axis=0)               # 4. batch of 1 -> (1,224,224,3)
    return arr
```

Inference then: `p = float(model.predict(arr, verbose=0)[0][0])`.

---

## 4. Design fidelity (very important)

The prototype `spotcheck-prototype.html` is the **source of truth for the
design**. The finished app must look **identical** to it. Do **not** redesign,
re-theme, or "improve" the layout, colors, fonts, spacing, or components.

- Keep the Google Fonts (`Fraunces`, `Hanken Grotesk`, `IBM Plex Mono`).
- Keep the CSS variables / color palette exactly (mint/teal theme, eczema =
  `#7A67A6`, tinea = `#1F8A9B`, etc.).
- The two condition photos (eczema, tinea) are **embedded as base64** inside the
  prototype HTML. They have been **decoded losslessly** into
  `app/static/img/eczema.jpg` and `app/static/img/tinea.jpg` and are referenced
  with `url_for`. The image bytes are identical to the prototype's (verified by
  sha256), so the visual result is unchanged; this only keeps `index.html`
  readable (~42 KB instead of ~246 KB) and lets the browser cache the photos.
- Preserve the existing client-side page navigation (`go()`), the accordions,
  the quick-nav, and the scroll-spy behavior. The four "pages" (Home & Scan,
  Eczema, Tinea, About Model) should behave exactly as in the prototype.
- Keep all educational copy and the About-Model numbers (dataset counts, metrics,
  confusion matrix) **verbatim** — they come from the real notebook. Do not
  invent, round differently, or alter any statistic.
- Content may be **added** as long as it is sourced, never invented: the eczema
  type definitions are condensed from Cleveland Clinic, and the About-Model
  figures (`app/static/img/model/`) plus the train/val/test loss and accuracy
  table come straight from the notebook. Every number on the page must be
  traceable to `versi-7-eczema-tinea-classification.md`; the test
  `test_about_shows_model_figures_and_metrics` pins them.

**Deliberate deviations from the prototype** (only these two; everything else is
verbatim):

1. **`.bar-fill{display:block;}` — a bug fix.** `.bar-fill` is a `<span>` with no
   `display` property, so it is an inline element — and inline elements ignore
   `width`/`height` per the CSS spec. The result/confidence bars therefore never
   rendered at all in the prototype (its fake 85/15 demo included). This is the
   only rule changed from the prototype's CSS.
2. **Scan flow additions in the dropzone — approved.** The idle dropzone is
   unchanged; these only apply once a photo is chosen:
   - **Preview.** The selected photo fills the dropzone so the user sees what is
     being analyzed.
   - **Scanning state.** While `/predict` is in flight, a scrim keeps the reticle
     and "Analyzing…" legible and a scan line sweeps the photo. The state is held
     for at least `MIN_SCAN_MS` (1200 ms, matching the `scan-sweep` keyframes) —
     prediction is usually faster than the animation can be read.
   - **Result state.** Once the bars are filled the scrim and overlay text are
     removed, so the photo is shown at full clarity.
   - **Scanning another photo.** A `.btn-ghost` button below the dropzone plus a
     hover hint over the photo. The button is what makes this discoverable on
     touch screens, which have no hover.

### The one behavioral change: real prediction instead of the fake demo

In the prototype, clicking the dropzone calls a fake `runDemo()` that hard-codes
85% / 15%. Replace this with a **real upload → predict flow**:

1. The dropzone accepts a real image file (click-to-browse **and** drag-and-drop),
   JPG or PNG, one image.
2. On selection, the file is sent to a Flask endpoint (`POST /predict`) as
   `multipart/form-data`.
3. The endpoint runs `preprocess` + `model.predict`, and returns JSON, e.g.:
   ```json
   { "verdict": "Eczema", "confidence": 85, "eczema_pct": 85, "tinea_pct": 15 }
   ```
4. The frontend fills in the verdict text, the confidence chip, and animates the
   Eczema/Tinea bars using the returned numbers (reuse the existing bar-fill
   animation — just feed it real values). Keep the "This is not a medical
   diagnosis" disclaimer visible.
5. Show a small loading state while the request is in flight, and a friendly
   error message if the upload isn't a valid image.

---

## 5. Target folder structure

```
SpotCheck-Flask/
├── app/
│   ├── __init__.py          # create_app() application factory
│   ├── routes.py            # "/" (page) and "/predict" (inference)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model_final_best.keras
│   │   └── inference.py     # load model once + letterbox + preprocess + predict
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css     # extracted from the prototype's <style> block
│   │   ├── js/
│   │   │   └── main.js       # extracted <script> + the real upload/predict logic
│   │   └── img/
│   │       ├── eczema.jpg    # decoded from the prototype's base64
│   │       ├── tinea.jpg     # decoded from the prototype's base64
│   │       └── samples/      # optional: a couple of test images
│   └── templates/
│       ├── base.html         # <head>, fonts, topbar, footer, {% block %}s
│       ├── index.html        # the four sections (home/scan, eczema, tinea, about)
│       └── (split further only if it stays faithful to the prototype)
├── tests/                    # pytest: conftest.py, test_inference.py, test_routes.py
├── config.py                 # Config / DevConfig / ProdConfig classes
├── requirements.txt          # production deps
├── requirements-dev.txt      # + pytest (not installed in the Docker image)
├── pytest.ini
├── run.py                    # entry point: app = create_app(); app.run()
├── .env                      # SECRET_KEY, MODEL_PATH, etc. (NOT committed)
├── .env.example              # template with placeholder values (committed)
├── .gitignore                # venv, __pycache__, .env, *.pyc
├── Dockerfile                # for deployment (added in the deployment phase)
├── CLAUDE.md
└── README.md
```

Because the education pages are static content driven by client-side navigation,
keep the app as **one served page** (`base.html` + `index.html`) with the
prototype's JS navigation intact — this is the most faithful and simplest
approach. Only split into more templates if the visual result stays identical.

---

## 6. Coding conventions

- Follow **PEP 8**. Keep functions small and named clearly.
- Use the **application factory** (`create_app()`) — no module-level global
  `app` object created at import time.
- Put configuration in `config.py` with separate classes for development and
  production; read secrets/paths from environment variables via `.env`
  (use `python-dotenv`). Never hard-code secrets.
- Keep **routes thin**: routes handle the HTTP request/response only; all model
  logic lives in `app/ml/inference.py`.
- **Load the model once** at startup, reuse it for every request.
- Validate uploads: check that a file was provided, that it's an allowed type
  (JPG/PNG), and cap the size (e.g. `MAX_CONTENT_LENGTH`). Return clear JSON
  errors rather than crashing.
- Comments and docstrings: concise, and it's fine to write them in Indonesian to
  match the thesis, as long as identifiers stay in English.
- Do **not** write or include any malicious, tracking, or data-collection code.
  Uploaded images are processed in memory for prediction and not stored
  permanently unless explicitly requested.

---

## 7. Deployment (later phase — don't start until the app runs locally)

Goal: make the app reachable by others, not just on `localhost`.

- Because the app depends on TensorFlow, the deployment target needs enough RAM
  (TF + model can use several hundred MB at load). Prefer a platform that gives
  comfortable memory on its free tier.
- **Recommended:** a **Docker-based deploy** (a `Dockerfile` running
  `gunicorn`), which works on **Hugging Face Spaces (Docker Space)** — ML-friendly,
  free, and generous on memory — or on **Render** as an alternative. Keep
  `requirements.txt` using `tensorflow-cpu` to shrink the image.
- Provide a `Dockerfile` and a short "How to deploy" section in `README.md` when
  this phase starts.
- Do not perform any deploy action, create accounts, or push to a host
  automatically — surface the commands/steps and let the developer run them.

---

## 8. Working process for Claude Code

1. **Before coding, produce a plan**: propose the folder structure and the build
   phases, and wait for approval.
2. Build in phases, checking in between:
   - **Phase 1** — Project skeleton: factory, `config.py`, `.env(.example)`,
     `requirements.txt`, `run.py`, `.gitignore`, and the templates/CSS/JS
     extracted from the prototype so the site renders exactly like the prototype
     (minus the proto-note banner, with "Streamlit" wording replaced).
   - **Phase 2** — Model integration: `app/ml/inference.py` (load once, letterbox
     preprocess, predict), the `/predict` route, and the real upload flow wired
     into the dropzone + result bars.
   - **Phase 3** — Polish: upload validation, loading/error states, README.
   - **Phase 4** — Deployment: `Dockerfile`, `tensorflow-cpu`, deploy notes.
3. Keep the code clean and the structure exactly as in section 5, so the
   implementation is easy to describe in the thesis (Bab 3).

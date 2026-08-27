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
looks more like **dermatitis** (non-infectious inflammation) or
**dermatophytosis** (a fungal infection), plus educational pages about each
group. It is explicitly **not** a medical diagnosis tool.

Both labels name a **group** of diseases, not a single one. Rather than write
a guide for every member, the education pages cover the best-known member of
each group — **eczema** for dermatitis, **ringworm** for dermatophytosis — and
say so explicitly in a `.scope-note` box on the home page and on each guide.
Never present the model's output as a prediction of eczema or ringworm
specifically.

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

- **Source notebook:** `v2-kl-r0-sds-dd-classification-44.ipynb` (root of the
  repo; the `.ipynb` itself is gitignored for size). This is the sole source of
  truth for every model number and every preprocessing step. The earlier
  `versi-7-eczema-tinea-classification.md` was deleted when the research moved
  from eczema/tinea to dermatitis/dermatophytosis — it survives only in git
  history (commit `2282767`) and must not be cited.
- **File:** `model_final_best.keras` under `app/ml/` — the notebook's export
  `v2-KL-r0-SDS-dermatitis-dermatophytosis-classification.keras`, renamed so the
  path stays stable across retrainings.
- **Config:** `app/ml/inference_config.json`, written by the notebook at export
  time. It records the class order, threshold and canvas; `inference.py`'s
  constants are pinned to it by a test.
- **Input shape:** `(224, 336, 3)` — a **336×224 landscape canvas (3:2)**, RGB,
  pixel values scaled to `0–1`. Note the tensor is (height, width, channels)
  while the PIL canvas tuple is (width, height).
- **Output:** a single **sigmoid** value in `[0, 1]` = **P(dermatophytosis)**.
- **Class mapping (from training):** `0 = dermatitis`, `1 = dermatophytosis`.
- **Decision threshold:** `p >= 0.5` → **Dermatophytosis**; `p < 0.5` →
  **Dermatitis**.
- **Confidence to display:** `max(p, 1 - p) * 100`, rounded.
- **Bar percentages to display:**
  - Dermatophytosis % = `round(p * 100)`
  - Dermatitis % = `round((1 - p) * 100)`

### Known limitation — the model has no "neither" answer

It is a **closed-set** binary classifier: one sigmoid, one threshold, two possible
answers. There is no rejection class and no out-of-distribution check, so any
image at all — healthy skin, another condition, a photo that isn't skin — still
comes back as one of the two, sometimes with high confidence. A gatekeeper model
was considered and not built.

Because a user can act on a wrong answer, this must be stated in the UI, not left
for the reader to infer. It currently appears in three places, pinned by
`test_index_warns_that_out_of_scope_photos_still_get_an_answer`:

1. a `.scope-note` after the "How it works" steps on the home page,
2. one clause in the result `.disclaimer`,
3. a technical `.scope-note` at the end of About §07.

Do not remove or soften these without replacing the guarantee some other way.

**Load the model once at application startup** (e.g. in the factory or an
`inference.py` module), never per request.

### Preprocessing — must match training EXACTLY

The model was trained on images that were first **rotated to landscape**, then
**letterbox-resized** onto a 336×224 canvas (proportional resize + centered
black padding), then normalized by dividing by 255. Reproduce this precisely for
every uploaded image — the rotation step is easy to miss and silently wrong.
Reference implementation, verbatim from the training notebook:

```python
from PIL import Image
import numpy as np

IMAGE_SIZE = (336, 224)          # (width, height) for PIL — a 3:2 landscape canvas

def canonical_orientation(img):
    # Portrait rotates to landscape; landscape and square pass through unchanged.
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

def preprocess(image_file):
    img = Image.open(image_file).convert("RGB")     # 1. force RGB
    img = canonical_orientation(img)                # 2. portrait -> landscape
    img = letterbox_resize(img, IMAGE_SIZE)         # 3. letterbox to 336x224
    arr = np.asarray(img, dtype="float32") / 255.0  # 4. normalize 0..1
    arr = np.expand_dims(arr, axis=0)               # 5. batch of 1 -> (1,224,336,3)
    return arr
```

Inference then: `p = float(model.predict(arr, verbose=0)[0][0])`.

The app adds one step ahead of all of these: `ImageOps.exif_transpose()`, so a
phone photo that stores its orientation as an EXIF tag is straightened before
the portrait/landscape decision is made. Training images carry no such tag, so
this changes nothing for training-style input.

---

## 4. Design fidelity (very important)

The prototype `spotcheck-prototype.html` is the **source of truth for the
design**. The finished app must look **identical** to it. Do **not** redesign,
re-theme, or "improve" the layout, colors, fonts, spacing, or components.

- Keep the Google Fonts (`Fraunces`, `Hanken Grotesk`, `IBM Plex Mono`).
- Keep the CSS variables / color palette exactly (mint/teal theme,
  `--dermatitis` = `#7A67A6`, `--dermatophytosis` = `#2A5FA8`, etc.). The
  variables were renamed from `--eczema`/`--tinea` when the classes changed;
  the colour values are unchanged.
- The two condition photos (eczema and ringworm — the example disease of each
  group) are **embedded as base64** inside the
  prototype HTML. They have been **decoded losslessly** into
  `app/static/img/eczema.jpg` and `app/static/img/tinea.jpg` and are referenced
  with `url_for`. The image bytes are identical to the prototype's (verified by
  sha256), so the visual result is unchanged; this only keeps `index.html`
  readable (~42 KB instead of ~246 KB) and lets the browser cache the photos.
- Preserve the existing client-side page navigation (`go()`), the accordions,
  the quick-nav, and the scroll-spy behavior. The four "pages" (Home & Scan,
  Dermatitis, Dermatophytosis, About Model) should behave exactly as in the
  prototype.
- Keep all educational copy and the About-Model numbers (dataset counts, metrics,
  confusion matrix) **verbatim** — they come from the real notebook. Do not
  invent, round differently, or alter any statistic.
- Content may be **added** as long as it is sourced, never invented: the eczema
  type definitions are condensed from Cleveland Clinic, and the About-Model
  figures (`app/static/img/model/`) plus the train/val/test metric table come
  straight from the notebook. Every number on the page must be traceable to
  `v2-kl-r0-sds-dd-classification-44.md`; the tests
  `test_about_shows_model_figures_and_metrics` and
  `test_about_reports_the_dataset_used_for_training` pin them.

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
   { "verdict": "Dermatitis", "confidence": 85,
     "dermatitis_pct": 85, "dermatophytosis_pct": 15 }
   ```
4. The frontend fills in the verdict text, the confidence chip, and animates the
   Dermatitis/Dermatophytosis bars using the returned numbers (reuse the bar-fill
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
│   │   ├── inference_config.json   # class order, threshold, canvas (from the notebook)
│   │   └── inference.py     # load model once + letterbox + preprocess + predict
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css     # extracted from the prototype's <style> block
│   │   ├── js/
│   │   │   └── main.js       # extracted <script> + the real upload/predict logic
│   │   └── img/
│   │       ├── eczema.jpg    # decoded from the prototype's base64
│   │       ├── tinea.jpg     # decoded from the prototype's base64
│   │       ├── favicon*.png / favicon.svg / apple-touch-icon.png
│   │       └── model/        # figures exported from the training notebook
│   └── templates/
│       ├── base.html         # <head>, fonts, topbar, footer, {% block %}s
│       ├── index.html        # four sections (home/scan, dermatitis, dermatophytosis, about)
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

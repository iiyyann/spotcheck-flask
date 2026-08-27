---
jupyter:
  kaggle:
    accelerator: none
    dataSources:
    - sourceId: 19130667
      sourceType: datasetVersion
    isGpuEnabled: false
    isInternetEnabled: true
    language: python
    sourceType: notebook
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.13
  nbformat: 4
  nbformat_minor: 5
  papermill:
    default_parameters: {}
    duration: 1753.058851
    end_time: "2026-08-25T22:31:59.627853+00:00"
    environment_variables: {}
    input_path: \_\_notebook\_\_.ipynb
    output_path: \_\_notebook\_\_.ipynb
    parameters: {}
    start_time: "2026-08-25T22:02:46.569002+00:00"
    version: 2.7.0
---

::: {#71f61a7a .cell .markdown papermill="{\"duration\":1.1808e-2,\"end_time\":\"2026-08-25T22:02:49.020854+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:02:49.009046+00:00\",\"status\":\"completed\"}" tags="[]"}
# 1. Pendahuluan {#1-pendahuluan}
:::

::: {#6adf2126 .cell .markdown papermill="{\"duration\":1.052e-2,\"end_time\":\"2026-08-25T22:02:49.042404+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:02:49.031884+00:00\",\"status\":\"completed\"}" tags="[]"}
# 2. Konfigurasi & Persiapan Lingkungan {#2-konfigurasi--persiapan-lingkungan}
:::

::: {#bcfefb0d .cell .code execution_count="1" execution="{\"iopub.execute_input\":\"2026-08-25T22:02:49.066589Z\",\"iopub.status.busy\":\"2026-08-25T22:02:49.066259Z\",\"iopub.status.idle\":\"2026-08-25T22:03:06.352181Z\",\"shell.execute_reply\":\"2026-08-25T22:03:06.351311Z\"}" papermill="{\"duration\":17.299764,\"end_time\":\"2026-08-25T22:03:06.353978+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:02:49.054214+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
import os

# Log informasi TensorFlow ditekan agar keluaran notebook tetap ringkas
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import json
import time
import random
import hashlib
import platform
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import imagehash
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_score, recall_score, f1_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint

# Seed global mengendalikan tahap yang harus dapat direproduksi persis,
# terutama pembagian data
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Seed pelatihan mengendalikan bobot awal, pengocokan, dan augmentasi
TRAINING_SEED = 44

# Ekstensi berkas yang dianggap sebagai citra valid
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Label kelas, terurut alfabetis: indeks 0 = dermatitis, indeks 1 = dermatophytosis
CLASS_NAMES = ["dermatitis", "dermatophytosis"]

# Kanvas target berbentuk 3:2 mengikuti rasio mayoritas citra
IMAGE_WIDTH = 336
IMAGE_HEIGHT = 224

# Urutan dimensi dipisahkan karena PIL dan TensorFlow berbeda konvensi
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)       # (lebar, tinggi) untuk PIL
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)   # (tinggi, lebar, kanal) untuk TensorFlow

# Konfigurasi pelatihan
EPOCHS = 150
BATCH_SIZE = 32

# Direktori dataset dan direktori kerja. Nama direktori kelas pada dataset
# berhuruf kapital di awal, sedangkan label kelas dipertahankan huruf kecil
BASE_DIR = Path("/kaggle/input/datasets/mreihandirizal/"
                "dermatitis-and-dermatophytosis-collection-pre-dedup")
DERMATITIS_DIR = BASE_DIR / "Dermatitis"
DERMATOPHYTOSIS_DIR = BASE_DIR / "Dermatophytosis"
WORKING_DIR = Path("/kaggle/working")

# Konfigurasi akselerator
strategy = tf.distribute.MirroredStrategy()
gpu_devices = tf.config.list_physical_devices("GPU")
accelerator = f"{len(gpu_devices)} GPU" if gpu_devices else "CPU (akselerator tidak aktif)"

# Gaya visualisasi seragam untuk seluruh notebook
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.titleweight": "bold",
})

# Versi pustaka utama untuk dokumentasi kebutuhan perangkat lunak
library_versions = {
    "Python": platform.python_version(),
    "NumPy": np.__version__,
    "pandas": pd.__version__,
    "Pillow (PIL)": PIL.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "scikit-learn": sklearn.__version__,
    "ImageHash": imagehash.__version__,
    "TensorFlow": tf.__version__,
}

W = 78
print("=" * W)
print("  KONFIGURASI LINGKUNGAN")
print("=" * W)
print(f"  {'Seed global':<30}: {SEED}")
print(f"  {'Seed pelatihan':<30}: {TRAINING_SEED}")
print("-" * W)
print(f"  {'Kelas (indeks 0 / 1)':<30}: {CLASS_NAMES[0]} / {CLASS_NAMES[1]}")
print(f"  {'Ekstensi citra valid':<30}: {', '.join(sorted(IMAGE_EXTENSIONS))}")
print("-" * W)
print(f"  {'Kanvas target (lebar x tinggi)':<30}: {IMAGE_WIDTH} x {IMAGE_HEIGHT} piksel")
print(f"  {'Rasio kanvas':<30}: {IMAGE_WIDTH / IMAGE_HEIGHT:.2f} : 1")
print(f"  {'Ukuran citra untuk PIL':<30}: {IMAGE_SIZE}")
print(f"  {'Bentuk masukan model':<30}: {INPUT_SHAPE}")
print("-" * W)
print(f"  {'Ukuran batch':<30}: {BATCH_SIZE}")
print(f"  {'Epoch maksimum':<30}: {EPOCHS}")
print(f"  {'Akselerator':<30}: {accelerator}")
print(f"  {'Replika strategi':<30}: {strategy.num_replicas_in_sync}")
print("-" * W)
print("  Direktori dataset")
print(f"    {BASE_DIR.parent}/")
print(f"      {BASE_DIR.name}/")
print("  Direktori kerja")
print(f"    {WORKING_DIR}")
print("-" * W)
print("  Versi Pustaka Utama")
print("-" * W)
for name, version in library_versions.items():
    print(f"  {name:<30}: {version}")
print("=" * W)

if not gpu_devices:
    print()
    print("  Peringatan: pelatihan akan berjalan pada CPU dan jauh lebih lambat.")
    print("  Setel Accelerator ke GPU pada panel Session options.")
```

::: {.output .stream .stdout}
    INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
    ==============================================================================
      KONFIGURASI LINGKUNGAN
    ==============================================================================
      Seed global                   : 42
      Seed pelatihan                : 44
    ------------------------------------------------------------------------------
      Kelas (indeks 0 / 1)          : dermatitis / dermatophytosis
      Ekstensi citra valid          : .bmp, .jpeg, .jpg, .png, .webp
    ------------------------------------------------------------------------------
      Kanvas target (lebar x tinggi): 336 x 224 piksel
      Rasio kanvas                  : 1.50 : 1
      Ukuran citra untuk PIL        : (336, 224)
      Bentuk masukan model          : (224, 336, 3)
    ------------------------------------------------------------------------------
      Ukuran batch                  : 32
      Epoch maksimum                : 150
      Akselerator                   : 1 GPU
      Replika strategi              : 1
    ------------------------------------------------------------------------------
      Direktori dataset
        /kaggle/input/datasets/mreihandirizal/
          dermatitis-and-dermatophytosis-collection-pre-dedup/
      Direktori kerja
        /kaggle/working
    ------------------------------------------------------------------------------
      Versi Pustaka Utama
    ------------------------------------------------------------------------------
      Python                        : 3.12.13
      NumPy                         : 2.0.2
      pandas                        : 2.3.3
      Pillow (PIL)                  : 11.3.0
      matplotlib                    : 3.10.0
      seaborn                       : 0.13.2
      scikit-learn                  : 1.6.1
      ImageHash                     : 4.3.2
      TensorFlow                    : 2.20.0
    ==============================================================================
:::

::: {.output .stream .stderr}
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1787695386.314299      23 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 15511 MB memory:  -> device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0, compute capability: 6.0
:::
:::

::: {#aa374934 .cell .markdown papermill="{\"duration\":1.0463e-2,\"end_time\":\"2026-08-25T22:03:06.375374+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:06.364911+00:00\",\"status\":\"completed\"}" tags="[]"}
# 3. Data Understanding {#3-data-understanding}
:::

::: {#a54eedd6 .cell .markdown papermill="{\"duration\":1.0834e-2,\"end_time\":\"2026-08-25T22:03:06.396853+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:06.386019+00:00\",\"status\":\"completed\"}" tags="[]"}
## 3.1 Pemeriksaan Struktur Dataset {#31-pemeriksaan-struktur-dataset}
:::

::: {#5c783347 .cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2026-08-25T22:03:06.420165Z\",\"iopub.status.busy\":\"2026-08-25T22:03:06.419317Z\",\"iopub.status.idle\":\"2026-08-25T22:03:25.187477Z\",\"shell.execute_reply\":\"2026-08-25T22:03:25.186406Z\"}" papermill="{\"duration\":18.78179,\"end_time\":\"2026-08-25T22:03:25.189151+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:06.407361+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def scan_dataset_structure():
    """Pindai kedua direktori kelas dan susun inventaris berkas.

    Pemindaian hanya membaca metadata sistem berkas. Isi citra belum
    dibuka pada tahap ini.
    """
    records, non_image_files, subdirectories = [], [], []

    for class_dir, label in [(DERMATITIS_DIR, "dermatitis"),
                             (DERMATOPHYTOSIS_DIR, "dermatophytosis")]:
        for path in sorted(class_dir.iterdir()):
            if path.is_dir():
                subdirectories.append(f"{class_dir.name}/{path.name}")
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                non_image_files.append(f"{class_dir.name}/{path.name}")
                continue

            records.append({
                "path": path,
                "filename": path.name,
                "label": label,
                "extension": path.suffix.lower(),
            })

    return pd.DataFrame(records), non_image_files, subdirectories


# Direktori wajib tersedia sebelum pemindaian dijalankan
missing_dirs = [d for d in (BASE_DIR, DERMATITIS_DIR, DERMATOPHYTOSIS_DIR)
                if not d.exists()]
if missing_dirs:
    raise FileNotFoundError(
        "Direktori berikut tidak ditemukan:\n  "
        + "\n  ".join(str(d) for d in missing_dirs)
    )

inventory_df, non_image_files, subdirectories = scan_dataset_structure()

n_dermatitis = int((inventory_df["label"] == "dermatitis").sum())
n_dermatophytosis = int((inventory_df["label"] == "dermatophytosis").sum())
n_total = len(inventory_df)

extension_summary = ", ".join(
    f"{extension} ({count})"
    for extension, count in inventory_df["extension"].value_counts().items()
)

W = 78
print("=" * W)
print("  STRUKTUR DATASET")
print("=" * W)
print(f"  {BASE_DIR.parent}/")
print(f"  └── {BASE_DIR.name}/")
print(f"      ├── {DERMATITIS_DIR.name + '/':<24}{n_dermatitis:>6} berkas")
print(f"      └── {DERMATOPHYTOSIS_DIR.name + '/':<24}{n_dermatophytosis:>6} berkas")
print("-" * W)
print(f"  {'Total berkas citra':<30}: {n_total}")
print(f"  {'Format berkas':<30}: {extension_summary}")
print(f"  {'Subdirektori bersarang':<30}: {len(subdirectories)}")
print(f"  {'Berkas non-citra':<30}: {len(non_image_files)}")
print("=" * W)

if subdirectories:
    print()
    print("  Subdirektori bersarang yang ditemukan:")
    for name in subdirectories[:10]:
        print(f"    {name}")
    if len(subdirectories) > 10:
        print(f"    ... dan {len(subdirectories) - 10} lainnya")

if non_image_files:
    print()
    print("  Berkas non-citra yang diabaikan:")
    for name in non_image_files[:10]:
        print(f"    {name}")
    if len(non_image_files) > 10:
        print(f"    ... dan {len(non_image_files) - 10} lainnya")
```

::: {.output .stream .stdout}
    ==============================================================================
      STRUKTUR DATASET
    ==============================================================================
      /kaggle/input/datasets/mreihandirizal/
      └── dermatitis-and-dermatophytosis-collection-pre-dedup/
          ├── Dermatitis/               5490 berkas
          └── Dermatophytosis/          4497 berkas
    ------------------------------------------------------------------------------
      Total berkas citra            : 9987
      Format berkas                 : .jpg (6481), .jpeg (3302), .png (204)
      Subdirektori bersarang        : 0
      Berkas non-citra              : 0
    ==============================================================================
:::
:::

::: {#6f887fd7 .cell .markdown papermill="{\"duration\":9.955e-3,\"end_time\":\"2026-08-25T22:03:25.209643+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:25.199688+00:00\",\"status\":\"completed\"}" tags="[]"}
## 3.2 Pemeriksaan Distribusi Kelas {#32-pemeriksaan-distribusi-kelas}
:::

::: {#3267e7a9 .cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2026-08-25T22:03:25.232289Z\",\"iopub.status.busy\":\"2026-08-25T22:03:25.23187Z\",\"iopub.status.idle\":\"2026-08-25T22:03:25.389883Z\",\"shell.execute_reply\":\"2026-08-25T22:03:25.389088Z\"}" papermill="{\"duration\":0.171167,\"end_time\":\"2026-08-25T22:03:25.391461+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:25.220294+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
class_counts = inventory_df["label"].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()

# Proporsi kedua kelas ditampilkan sebagai satu batang bertumpuk agar
# pergeseran sekat terhadap titik seimbang 50% terlihat langsung
class_colors = ["#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(9, 2.3))
offset = 0.0

for class_name, color in zip(CLASS_NAMES, class_colors):
    count = int(class_counts[class_name])
    percentage = count / n_total * 100
    ax.barh(0, percentage, left=offset, height=0.5, color=color)
    ax.text(offset + percentage / 2, 0,
            f"{class_name}\n{count} ({percentage:.2f}%)",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    offset += percentage

ax.axvline(50, color="#333333", linestyle="--", linewidth=1.2)
ax.text(50, 0.44, "Titik seimbang (50%)", ha="center", va="bottom",
        fontsize=9, color="#333333",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

ax.set_title("Distribusi Kelas pada Dataset", fontsize=12, pad=32)
ax.text(0.5, 1.07,
        f"Total {n_total} citra   |   Rasio = {imbalance_ratio:.2f} : 1",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color="#444444")

ax.set_xlim(0, 100)
ax.set_ylim(-0.45, 0.68)
ax.set_xlabel("Proporsi terhadap total citra (%)")
ax.set_yticks([])
ax.grid(False)
ax.spines[["top", "right", "left"]].set_visible(False)
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/9d68533758aad0b924d73173c31f3559d867c7a2.png)
:::
:::

::: {#eac726cc .cell .markdown papermill="{\"duration\":1.0986e-2,\"end_time\":\"2026-08-25T22:03:25.413585+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:25.402599+00:00\",\"status\":\"completed\"}" tags="[]"}
## 3.3 Pemeriksaan Distribusi Sumber Data {#33-pemeriksaan-distribusi-sumber-data}
:::

::: {#7250d27f .cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2026-08-25T22:03:25.436463Z\",\"iopub.status.busy\":\"2026-08-25T22:03:25.435856Z\",\"iopub.status.idle\":\"2026-08-25T22:03:25.690681Z\",\"shell.execute_reply\":\"2026-08-25T22:03:25.689747Z\"}" papermill="{\"duration\":0.268357,\"end_time\":\"2026-08-25T22:03:25.692291+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:25.423934+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
# Kode sumber A-G mengikuti urutan unggah di Kaggle, sama dengan penetapan
# pada notebook pengumpulan data
SOURCE_MAP = {
    "A": "DermNet",
    "B": "Skin Disease Image Dataset",
    "C": "Ringworm Dataset for Classification",
    "D": "31 Classes of Skin Disease",
    "E": "Skin Diseases Cancer Comprehensive Dataset",
    "F": "Skin Disease Curated Dataset",
    "G": "Tinea Clean DermNet Roboflow",
}


def extract_source_code(stem):
    """Ambil kode sumber (A-G) dari prefiks nama berkas."""
    match = re.match(r"^([A-G])_", stem)
    return match.group(1) if match else "tidak dikenali"


inventory_df["source"] = inventory_df["path"].map(lambda p: extract_source_code(p.stem))

source_table = (inventory_df.groupby(["source", "label"]).size()
                .unstack(fill_value=0)
                .reindex(index=list(SOURCE_MAP), columns=CLASS_NAMES, fill_value=0))

class_colors = ["#4C72B0", "#DD8452"]
bar_height = 0.38
max_value = int(source_table.to_numpy().max())
label_gap = max_value * 0.012
positions = np.arange(len(source_table))

fig, ax = plt.subplots(figsize=(10, 5.8))

for index, (class_name, color) in enumerate(zip(CLASS_NAMES, class_colors)):
    values = source_table[class_name].to_numpy()
    offsets = positions + (bar_height / 2 if index == 0 else -bar_height / 2)
    ax.barh(offsets, values, height=bar_height, color=color, label=class_name)

    # Sumber tanpa kontribusi dibiarkan tanpa batang, namun angka nolnya
    # tetap ditulis pada titik awal sumbu
    for offset, value in zip(offsets, values):
        ax.text(value + label_gap, offset, f"{int(value)}",
                va="center", fontsize=9,
                color="#444444" if value == 0 else "black")

ax.set_title("Distribusi Sumber Data pada Dataset", fontsize=12, pad=30)
ax.text(0.5, 1.04,
        f"{len(SOURCE_MAP)} repositori Kaggle   |   Total {n_total} citra",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color="#444444")

ax.set_yticks(positions)
ax.set_yticklabels([f"{code}.   {SOURCE_MAP[code]}" for code in source_table.index],
                   fontsize=9.5)
ax.set_xlabel("Jumlah citra")
ax.set_xlim(0, max_value * 1.1)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(loc="lower right", frameon=True, fontsize=9.5)
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/bddcad6b430e0caa2539822dc9f1ad30a67924da.png)
:::
:::

::: {#d94cf5a0 .cell .markdown papermill="{\"duration\":1.1037e-2,\"end_time\":\"2026-08-25T22:03:25.714901+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:25.703864+00:00\",\"status\":\"completed\"}" tags="[]"}
## 3.4 Pemeriksaan Integritas Citra {#34-pemeriksaan-integritas-citra}
:::

::: {#67d86659 .cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2026-08-25T22:03:25.739665Z\",\"iopub.status.busy\":\"2026-08-25T22:03:25.73934Z\",\"iopub.status.idle\":\"2026-08-25T22:04:03.294773Z\",\"shell.execute_reply\":\"2026-08-25T22:04:03.293913Z\"}" papermill="{\"duration\":37.58339,\"end_time\":\"2026-08-25T22:04:03.309757+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:03:25.726367+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def inspect_image_integrity(inventory):
    """Buka setiap berkas untuk memastikan citra dapat dibaca utuh.

    Dimensi dan mode warna ikut dikumpulkan pada pemeriksaan ini agar
    berkas tidak perlu dibuka ulang pada sub bagian berikutnya.
    """
    records = []

    for path in inventory["path"]:
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                # Memeriksa kerusakan berkas tanpa mendekode citra penuh
                img.verify()
            records.append({
                "readable": True,
                "width": width,
                "height": height,
                "mode": mode,
                "error": None,
            })
        except Exception as error:
            records.append({
                "readable": False,
                "width": np.nan,
                "height": np.nan,
                "mode": None,
                "error": type(error).__name__,
            })

    return pd.concat(
        [inventory.reset_index(drop=True), pd.DataFrame(records)], axis=1
    )


inspected_df = inspect_image_integrity(inventory_df)

# Hanya citra utuh yang diteruskan ke seluruh tahap berikutnya
image_df = inspected_df[inspected_df["readable"]].reset_index(drop=True)

n_inspected = len(inspected_df)
n_intact = len(image_df)
n_corrupt = n_inspected - n_intact

W = 78
print("=" * W)
print("  INTEGRITAS CITRA")
print("=" * W)
print(f"  {'Berkas diperiksa':<30}: {n_inspected}")
print(f"  {'Citra utuh':<30}: {n_intact}")
print(f"  {'Citra rusak':<30}: {n_corrupt}")
print("=" * W)

if n_corrupt:
    corrupt_df = inspected_df[~inspected_df["readable"]]

    print()
    print("  Citra rusak per kelas:")
    for class_name in CLASS_NAMES:
        print(f"    {class_name:<28}: {int((corrupt_df['label'] == class_name).sum())}")

    print()
    print("  Daftar citra rusak:")
    print(f"    {'Kelas':<18}{'Berkas':<44}{'Penyebab'}")
    for _, row in corrupt_df.head(10).iterrows():
        filename = row["filename"]
        if len(filename) > 42:
            filename = filename[:39] + "..."
        print(f"    {row['label']:<18}{filename:<44}{row['error']}")
    if n_corrupt > 10:
        print(f"    ... dan {n_corrupt - 10} lainnya")
```

::: {.output .stream .stdout}
    ==============================================================================
      INTEGRITAS CITRA
    ==============================================================================
      Berkas diperiksa              : 9987
      Citra utuh                    : 9987
      Citra rusak                   : 0
    ==============================================================================
:::
:::

::: {#cbbb4ae4 .cell .markdown papermill="{\"duration\":1.2528e-2,\"end_time\":\"2026-08-25T22:04:03.335006+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:03.322478+00:00\",\"status\":\"completed\"}" tags="[]"}
## 3.5 Pemeriksaan Mode Warna Citra {#35-pemeriksaan-mode-warna-citra}
:::

::: {#f8091b12 .cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:03.362379Z\",\"iopub.status.busy\":\"2026-08-25T22:04:03.361958Z\",\"iopub.status.idle\":\"2026-08-25T22:04:03.373053Z\",\"shell.execute_reply\":\"2026-08-25T22:04:03.372228Z\"}" papermill="{\"duration\":2.6317e-2,\"end_time\":\"2026-08-25T22:04:03.374629+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:03.348312+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
# Mode PIL yang menyimpan citra tanpa informasi warna
GRAYSCALE_MODES = {"L", "LA", "1", "I", "F"}


def categorize_color_mode(mode):
    """Kelompokkan mode PIL menurut kebutuhan konversi ke RGB."""
    if mode == "RGB":
        return "RGB"
    if mode in GRAYSCALE_MODES:
        return "skala keabuan"
    return "mode lain"


image_df["color_group"] = image_df["mode"].map(categorize_color_mode)

group_counts = image_df["color_group"].value_counts()
n_non_rgb = n_intact - int(group_counts.get("RGB", 0))

W = 78
print("=" * W)
print("  MODE WARNA CITRA")
print("=" * W)
for group_name in ["RGB", "skala keabuan", "mode lain"]:
    print(f"  {group_name:<30}: {int(group_counts.get(group_name, 0))}")
print("-" * W)
print(f"  {'Citra perlu konversi RGB':<30}: {n_non_rgb}")
print("=" * W)

if n_non_rgb:
    non_rgb_counts = (image_df[image_df["color_group"] != "RGB"]["mode"]
                      .value_counts())
    print()
    print("  Rincian mode citra non-RGB:")
    for mode_name, count in non_rgb_counts.items():
        print(f"    {str(mode_name):<28}: {count}")
```

::: {.output .stream .stdout}
    ==============================================================================
      MODE WARNA CITRA
    ==============================================================================
      RGB                           : 9987
      skala keabuan                 : 0
      mode lain                     : 0
    ------------------------------------------------------------------------------
      Citra perlu konversi RGB      : 0
    ==============================================================================
:::
:::

::: {#e0b01762 .cell .markdown papermill="{\"duration\":1.1192e-2,\"end_time\":\"2026-08-25T22:04:03.398212+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:03.38702+00:00\",\"status\":\"completed\"}" tags="[]"}
## 3.6 Pemeriksaan Geometri Citra {#36-pemeriksaan-geometri-citra}
:::

::: {#7d416220 .cell .code execution_count="7" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:03.423366Z\",\"iopub.status.busy\":\"2026-08-25T22:04:03.422729Z\",\"iopub.status.idle\":\"2026-08-25T22:04:04.146708Z\",\"shell.execute_reply\":\"2026-08-25T22:04:04.145785Z\"}" papermill="{\"duration\":0.738919,\"end_time\":\"2026-08-25T22:04:04.148291+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:03.409372+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
image_df["long_side"] = image_df[["width", "height"]].max(axis=1)
image_df["short_side"] = image_df[["width", "height"]].min(axis=1)
image_df["aspect"] = image_df["long_side"] / image_df["short_side"]

# Citra dengan sisi sama panjang dipisahkan tersendiri karena tidak
# memiliki orientasi yang perlu dikanonisasi
image_df["orientation"] = np.select(
    [image_df["height"] > image_df["width"],
     image_df["height"] < image_df["width"]],
    ["portrait", "landscape"],
    default="persegi"
)

ORIENTATIONS = ["landscape", "portrait", "persegi"]

aspect = image_df["aspect"].to_numpy()
short_side = image_df["short_side"].to_numpy()
long_side = image_df["long_side"].to_numpy()
canvas_ratio = IMAGE_WIDTH / IMAGE_HEIGHT

W = 78
print("=" * W)
print("  GEOMETRI CITRA")
print("=" * W)
print("  Orientasi")
for orientation_name in ORIENTATIONS:
    mask = image_df["orientation"] == orientation_name
    print(f"    {orientation_name:<28}: {int(mask.sum())} ({mask.mean() * 100:.2f}%)")
print("-" * W)
print("  Ukuran Piksel")
print(f"    {'':<16}{'Minimum':>10}{'Median':>10}{'Maksimum':>10}")
print(f"    {'Sisi pendek':<16}{short_side.min():>10.0f}"
      f"{np.median(short_side):>10.0f}{short_side.max():>10.0f}")
print(f"    {'Sisi panjang':<16}{long_side.min():>10.0f}"
      f"{np.median(long_side):>10.0f}{long_side.max():>10.0f}")
print("-" * W)
print("  Rasio Aspek (sisi panjang / sisi pendek)")
print(f"    {'Persentil 10':<28}: {np.percentile(aspect, 10):.2f}")
print(f"    {'Median':<28}: {np.median(aspect):.2f}")
print(f"    {'Persentil 90':<28}: {np.percentile(aspect, 90):.2f}")
print(f"    {'Rasio kanvas target':<28}: {canvas_ratio:.2f}")
print("=" * W)

# Sebaran rasio aspek terhadap dua acuan: bentuk persegi dan rasio kanvas
# target, sebagai dasar penetapan kanvas pada tahap transformasi
bins = np.linspace(1.0, np.percentile(aspect, 99.5), 80)
class_colors = ["#4C72B0", "#DD8452"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
fig.suptitle("Sebaran Rasio Aspek Citra", fontsize=12)

for ax, class_name, color in zip(axes, CLASS_NAMES, class_colors):
    subset = image_df.loc[image_df["label"] == class_name, "aspect"]
    ax.hist(subset, bins=bins, color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="#55A868", linestyle=":", linewidth=1.6,
               label="Persegi (1.00)")
    ax.axvline(canvas_ratio, color="#333333", linestyle="--", linewidth=1.4,
               label=f"Kanvas target ({canvas_ratio:.2f})")
    ax.set_title(f"{class_name} (n = {len(subset)})", fontsize=10.5,
                 fontweight="normal")
    ax.set_xlabel("Sisi panjang / sisi pendek")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Jumlah citra")
axes[0].legend(fontsize=9)
plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    ==============================================================================
      GEOMETRI CITRA
    ==============================================================================
      Orientasi
        landscape                   : 7287 (72.96%)
        portrait                    : 2686 (26.89%)
        persegi                     : 14 (0.14%)
    ------------------------------------------------------------------------------
      Ukuran Piksel
                           Minimum    Median  Maksimum
        Sisi pendek             72       477      1753
        Sisi panjang            81       720      2337
    ------------------------------------------------------------------------------
      Rasio Aspek (sisi panjang / sisi pendek)
        Persentil 10                : 1.47
        Median                      : 1.51
        Persentil 90                : 1.53
        Rasio kanvas target         : 1.50
    ==============================================================================
:::

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/d8e9491b7cc09561654454fea848b516da977a1f.png)
:::
:::

::: {#dfefd087 .cell .markdown papermill="{\"duration\":1.1881e-2,\"end_time\":\"2026-08-25T22:04:04.172597+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:04.160716+00:00\",\"status\":\"completed\"}" tags="[]"}
# 4. Data Preprocessing {#4-data-preprocessing}
:::

::: {#4f461cdc .cell .markdown papermill="{\"duration\":1.3216e-2,\"end_time\":\"2026-08-25T22:04:04.199093+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:04.185877+00:00\",\"status\":\"completed\"}" tags="[]"}
## 4.1 Pembersihan Citra Duplikat {#41-pembersihan-citra-duplikat}
:::

::: {#a3d0e282 .cell .markdown papermill="{\"duration\":1.2401e-2,\"end_time\":\"2026-08-25T22:04:04.223586+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:04.211185+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.1.1 Duplikat Identik (Exact Duplicate) {#411-duplikat-identik-exact-duplicate}
:::

::: {#01945f1d .cell .markdown papermill="{\"duration\":1.2171e-2,\"end_time\":\"2026-08-25T22:04:04.24792+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:04.235749+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.1.1.1 Pemeriksaan Duplikat Identik {#4111-pemeriksaan-duplikat-identik}
:::

::: {#5869208f .cell .code execution_count="8" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:04.274238Z\",\"iopub.status.busy\":\"2026-08-25T22:04:04.273803Z\",\"iopub.status.idle\":\"2026-08-25T22:04:11.230656Z\",\"shell.execute_reply\":\"2026-08-25T22:04:11.229789Z\"}" papermill="{\"duration\":6.97214,\"end_time\":\"2026-08-25T22:04:11.232385+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:04.260245+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def compute_md5(path):
    """Hitung sidik jari MD5 dari isi berkas."""
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


image_df["md5"] = [compute_md5(path) for path in image_df["path"]]

n_unique_md5 = image_df["md5"].nunique()
n_duplicated_images = int(image_df.duplicated(subset="md5", keep=False).sum())
n_singleton_images = len(image_df) - n_duplicated_images

group_sizes = image_df.groupby("md5").size()
duplicate_group_sizes = group_sizes[group_sizes > 1]
duplicate_hashes = duplicate_group_sizes.index

# Grup yang anggotanya berasal dari lebih dari satu sumber atau kelas
group_source_count = image_df.groupby("md5")["source"].nunique()
group_class_count = image_df.groupby("md5")["label"].nunique()

cross_source_hashes = group_source_count[group_source_count > 1].index
cross_class_hashes = group_class_count[group_class_count > 1].index

W = 78
print("=" * W)
print("  DUPLIKAT IDENTIK")
print("=" * W)
print(f"  {'Total citra diperiksa':<34}: {len(image_df)}")
print(f"  {'Sidik jari MD5 unik':<34}: {n_unique_md5}")
print("-" * W)
print(f"  {'Citra yang memiliki kembaran':<34}: {n_duplicated_images}")
print(f"  {'Citra tanpa kembaran':<34}: {n_singleton_images}")
print(f"  {'Grup duplikat':<34}: {len(duplicate_hashes)}")
print(f"  {'Anggota tersedikit dalam satu grup':<34}: "
      f"{int(duplicate_group_sizes.min())}")
print(f"  {'Anggota terbanyak dalam satu grup':<34}: "
      f"{int(duplicate_group_sizes.max())}")
print("-" * W)
print(f"  {'Grup lintas sumber':<34}: {len(cross_source_hashes)}")
print(f"  {'Grup lintas kelas':<34}: {len(cross_class_hashes)}")
print("=" * W)

if len(cross_class_hashes) > 0:
    cross_class_df = image_df[image_df["md5"].isin(cross_class_hashes)]

    print()
    print("  Citra identik yang muncul pada kedua kelas:")
    for md5_value in list(cross_class_hashes)[:5]:
        subset = cross_class_df[cross_class_df["md5"] == md5_value]
        print(f"    {md5_value[:16]}...")
        for class_name in CLASS_NAMES:
            sources = sorted(set(subset.loc[subset["label"] == class_name, "source"]))
            if sources:
                print(f"      {class_name:<20}: sumber {', '.join(sources)}")
    if len(cross_class_hashes) > 5:
        print(f"    ... dan {len(cross_class_hashes) - 5} grup lainnya")
```

::: {.output .stream .stdout}
    ==============================================================================
      DUPLIKAT IDENTIK
    ==============================================================================
      Total citra diperiksa             : 9987
      Sidik jari MD5 unik               : 4296
    ------------------------------------------------------------------------------
      Citra yang memiliki kembaran      : 8255
      Citra tanpa kembaran              : 1732
      Grup duplikat                     : 2564
      Anggota tersedikit dalam satu grup: 2
      Anggota terbanyak dalam satu grup : 13
    ------------------------------------------------------------------------------
      Grup lintas sumber                : 2536
      Grup lintas kelas                 : 0
    ==============================================================================
:::
:::

::: {#f54fa8b4 .cell .markdown papermill="{\"duration\":1.4116e-2,\"end_time\":\"2026-08-25T22:04:11.260471+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:11.246355+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.1.1.2 Sampel Duplikat Identik {#4112-sampel-duplikat-identik}
:::

::: {#90508301 .cell .code execution_count="9" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:11.290109Z\",\"iopub.status.busy\":\"2026-08-25T22:04:11.289726Z\",\"iopub.status.idle\":\"2026-08-25T22:04:12.04343Z\",\"shell.execute_reply\":\"2026-08-25T22:04:12.042465Z\"}" papermill="{\"duration\":0.775259,\"end_time\":\"2026-08-25T22:04:12.049928+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:11.274669+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
SAMPLE_GROUPS = 3
SAMPLE_MEMBERS = 4
NAME_LIMIT = 24


def shorten_filename(filename, limit=NAME_LIMIT):
    """Pendekkan nama berkas agar muat sebagai keterangan gambar."""
    return filename if len(filename) <= limit else filename[:limit - 3] + "..."


if len(duplicate_hashes) == 0:
    print("Tidak ditemukan duplikat identik.")
else:
    # Grup dengan anggota terbanyak dipilih sebagai contoh karena paling
    # jelas memperlihatkan peredaran citra yang sama antar sumber
    sample_hashes = (duplicate_group_sizes
                     .sort_values(ascending=False)
                     .head(SAMPLE_GROUPS)
                     .index)

    fig, axes = plt.subplots(
        len(sample_hashes), SAMPLE_MEMBERS,
        figsize=(2.7 * SAMPLE_MEMBERS, 2.6 * len(sample_hashes))
    )
    axes = np.atleast_2d(axes)

    for row, md5_value in enumerate(sample_hashes):
        members = (image_df[image_df["md5"] == md5_value]
                   .sort_values("filename")
                   .reset_index(drop=True))

        for col in range(SAMPLE_MEMBERS):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col >= len(members):
                ax.set_visible(False)
                continue

            member = members.loc[col]
            with Image.open(member["path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"Sumber {member['source']}\n"
                         f"{shorten_filename(member['filename'])}",
                         fontsize=7.5, pad=4)

        axes[row][0].set_ylabel(f"Grup {row + 1}\n{len(members)} anggota",
                                fontsize=9.5)

    fig.suptitle(f"Contoh Grup Duplikat Identik "
                 f"(maksimal {SAMPLE_MEMBERS} anggota ditampilkan per grup)",
                 fontsize=12)
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/954da728de326452c859229a8bf06605682442be.png)
:::
:::

::: {#0fd572cb .cell .markdown papermill="{\"duration\":1.7225e-2,\"end_time\":\"2026-08-25T22:04:12.085338+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:12.068113+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.1.1.3 Penghapusan Duplikat Identik {#4113-penghapusan-duplikat-identik}
:::

::: {#ae9f4018 .cell .code execution_count="10" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:12.12016Z\",\"iopub.status.busy\":\"2026-08-25T22:04:12.119692Z\",\"iopub.status.idle\":\"2026-08-25T22:04:12.150541Z\",\"shell.execute_reply\":\"2026-08-25T22:04:12.149554Z\"}" papermill="{\"duration\":5.0807e-2,\"end_time\":\"2026-08-25T22:04:12.152277+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:12.10147+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def format_class_ratio(counts):
    """Susun rasio antar kelas dengan kelas kedua sebagai acuan 1.00."""
    values = np.array([counts.get(name, 0) for name in CLASS_NAMES], dtype=float)
    scaled = values / values[-1]
    return " : ".join(f"{value:.2f}" for value in scaled)


before_counts = image_df["label"].value_counts()

# Anggota yang dipertahankan ditetapkan dari kode sumber terkecil, lalu nama
# berkas terkecil, agar hasil penghapusan bersifat deterministik
exact_unique_df = (image_df
                   .sort_values(["source", "filename"])
                   .drop_duplicates(subset="md5", keep="first")
                   .sort_index()
                   .reset_index(drop=True))

n_before_exact = len(image_df)
n_after_exact = len(exact_unique_df)
n_removed_exact = n_before_exact - n_after_exact

after_counts = exact_unique_df["label"].value_counts()

W = 78
print("=" * W)
print("  PENGHAPUSAN DUPLIKAT IDENTIK")
print("=" * W)
print(f"  {'Citra sebelum':<34}: {n_before_exact} (100.00%)")
print(f"  {'Citra dihapus':<34}: {n_removed_exact} "
      f"({n_removed_exact / n_before_exact * 100:.2f}%)")
print(f"  {'Citra tersisa':<34}: {n_after_exact} "
      f"({n_after_exact / n_before_exact * 100:.2f}%)")
print("-" * W)
print("  Komposisi Kelas")
print(f"    {'Kelas':<26}{'Sebelum':>7}{'Dihapus':>10}{'Sesudah':>10}")
for class_name in CLASS_NAMES:
    n_class_before = int(before_counts.get(class_name, 0))
    n_class_after = int(after_counts.get(class_name, 0))
    print(f"    {class_name:<26}{n_class_before:>7}"
          f"{n_class_before - n_class_after:>10}{n_class_after:>10}")
print("-" * W)
print(f"  Rasio {CLASS_NAMES[0]} : {CLASS_NAMES[1]} = "
      f"{format_class_ratio(after_counts)} "
      f"(sebelumnya {format_class_ratio(before_counts)})")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PENGHAPUSAN DUPLIKAT IDENTIK
    ==============================================================================
      Citra sebelum                     : 9987 (100.00%)
      Citra dihapus                     : 5691 (56.98%)
      Citra tersisa                     : 4296 (43.02%)
    ------------------------------------------------------------------------------
      Komposisi Kelas
        Kelas                     Sebelum   Dihapus   Sesudah
        dermatitis                   5490      3517      1973
        dermatophytosis              4497      2174      2323
    ------------------------------------------------------------------------------
      Rasio dermatitis : dermatophytosis = 0.85 : 1.00 (sebelumnya 1.22 : 1.00)
    ==============================================================================
:::
:::

::: {#3bbfc27c .cell .markdown papermill="{\"duration\":1.8894e-2,\"end_time\":\"2026-08-25T22:04:12.18883+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:12.169936+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.1.2 Duplikat Serupa (Near Duplicate) {#412-duplikat-serupa-near-duplicate}
:::

::: {#7dbfd9e7 .cell .markdown papermill="{\"duration\":1.7699e-2,\"end_time\":\"2026-08-25T22:04:12.223624+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:12.205925+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.1.2.1 Pemeriksaan Duplikat Serupa {#4121-pemeriksaan-duplikat-serupa}
:::

::: {#21336868 .cell .code execution_count="11" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:12.2577Z\",\"iopub.status.busy\":\"2026-08-25T22:04:12.256872Z\",\"iopub.status.idle\":\"2026-08-25T22:04:36.909561Z\",\"shell.execute_reply\":\"2026-08-25T22:04:36.908564Z\"}" papermill="{\"duration\":24.671771,\"end_time\":\"2026-08-25T22:04:36.911409+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:12.239638+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
# Batas jarak yang didata sengaja lebih longgar daripada ambang penghapusan
# agar sebaran di sekitar ambang ikut terlihat
PHASH_SCAN_LIMIT = 12


def compute_phash(path):
    """Hitung sidik jari perseptual 64 bit dari sebuah citra."""
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


exact_unique_df["phash"] = [compute_phash(path) for path in exact_unique_df["path"]]

# Sidik jari disusun sebagai matriks bit agar jarak Hamming dapat dihitung
# secara vektor, satu citra terhadap seluruh citra sesudahnya
hash_bits = np.array([value.hash.flatten() for value in exact_unique_df["phash"]])
labels_array = exact_unique_df["label"].to_numpy()

index_a, index_b, distance_values = [], [], []
for i in range(len(hash_bits) - 1):
    distances = np.count_nonzero(hash_bits[i] != hash_bits[i + 1:], axis=1)
    matches = np.flatnonzero(distances <= PHASH_SCAN_LIMIT)
    if matches.size == 0:
        continue
    index_a.append(np.full(matches.size, i))
    index_b.append(matches + i + 1)
    distance_values.append(distances[matches])

pairs_df = pd.DataFrame({
    "idx_a": np.concatenate(index_a) if index_a else np.empty(0, dtype=int),
    "idx_b": np.concatenate(index_b) if index_b else np.empty(0, dtype=int),
    "distance": (np.concatenate(distance_values) if distance_values
                 else np.empty(0, dtype=int)),
})
pairs_df["same_class"] = (labels_array[pairs_df["idx_a"].to_numpy()]
                          == labels_array[pairs_df["idx_b"].to_numpy()])

same_class_pairs = pairs_df[pairs_df["same_class"]]
cross_class_pairs = pairs_df[~pairs_df["same_class"]]

n_scanned = len(exact_unique_df)
n_pairs_compared = n_scanned * (n_scanned - 1) // 2

W = 78
print("=" * W)
print("  DUPLIKAT SERUPA")
print("=" * W)
print(f"  {'Citra diperiksa':<34}: {n_scanned}")
print(f"  {'Pasangan dibandingkan':<34}: {n_pairs_compared}")
print(f"  {'Batas jarak yang didata':<34}: <= {PHASH_SCAN_LIMIT}")
print("-" * W)
print(f"  {'Pasangan dalam batas':<34}: {len(pairs_df)}")
print(f"  {'Pasangan sekelas':<34}: {len(same_class_pairs)}")
print(f"  {'Pasangan beda kelas':<34}: {len(cross_class_pairs)}")
print("-" * W)
print("  Sebaran Jarak Hamming")
print(f"    {'Jarak':<12}{'Sekelas':>10}{'Beda kelas':>13}{'Kumulatif':>12}")
cumulative = 0
for distance_value in range(0, PHASH_SCAN_LIMIT + 1, 2):
    n_same = int((same_class_pairs["distance"] == distance_value).sum())
    n_cross = int((cross_class_pairs["distance"] == distance_value).sum())
    cumulative += n_same + n_cross
    print(f"    {distance_value:<12}{n_same:>10}{n_cross:>13}{cumulative:>12}")
print("-" * W)
print("  Jarak ganjil tidak muncul karena setiap sidik jari selalu memiliki")
print("  tepat 32 bit bernilai satu, sehingga perubahan bit selalu berpasangan.")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      DUPLIKAT SERUPA
    ==============================================================================
      Citra diperiksa                   : 4296
      Pasangan dibandingkan             : 9225660
      Batas jarak yang didata           : <= 12
    ------------------------------------------------------------------------------
      Pasangan dalam batas              : 1656
      Pasangan sekelas                  : 1507
      Pasangan beda kelas               : 149
    ------------------------------------------------------------------------------
      Sebaran Jarak Hamming
        Jarak          Sekelas   Beda kelas   Kumulatif
        0                  857            0         857
        2                  140            1         998
        4                   40            1        1039
        6                   60            2        1101
        8                   80            2        1183
        10                 101           27        1311
        12                 229          116        1656
    ------------------------------------------------------------------------------
      Jarak ganjil tidak muncul karena setiap sidik jari selalu memiliki
      tepat 32 bit bernilai satu, sehingga perubahan bit selalu berpasangan.
    ==============================================================================
:::
:::

::: {#85c24a53 .cell .markdown papermill="{\"duration\":1.6353e-2,\"end_time\":\"2026-08-25T22:04:36.945074+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:36.928721+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.1.2.2 Sampel Duplikat Serupa Sekelas {#4122-sampel-duplikat-serupa-sekelas}
:::

::: {#05caf8c0 .cell .code execution_count="12" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:36.978977Z\",\"iopub.status.busy\":\"2026-08-25T22:04:36.978281Z\",\"iopub.status.idle\":\"2026-08-25T22:04:38.181242Z\",\"shell.execute_reply\":\"2026-08-25T22:04:38.180169Z\"}" papermill="{\"duration\":1.252864,\"end_time\":\"2026-08-25T22:04:38.213823+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:36.960959+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
if len(same_class_pairs) == 0:
    print("Tidak ditemukan duplikat serupa sekelas.")
else:
    # Satu pasangan contoh untuk setiap tingkat jarak agar peluruhan
    # kemiripan terhadap jarak Hamming dapat diamati langsung
    distance_levels = [value for value in range(0, PHASH_SCAN_LIMIT + 1, 2)
                       if (same_class_pairs["distance"] == value).any()]

    fig, axes = plt.subplots(len(distance_levels), 2,
                             figsize=(6.8, 2.8 * len(distance_levels)))
    axes = np.atleast_2d(axes)

    for row, distance_value in enumerate(distance_levels):
        pair = (same_class_pairs[same_class_pairs["distance"] == distance_value]
                .iloc[0])

        for col, index in enumerate([int(pair["idx_a"]), int(pair["idx_b"])]):
            member = exact_unique_df.loc[index]
            ax = axes[row][col]
            with Image.open(member["path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"Sumber {member['source']}\n"
                         f"{shorten_filename(member['filename'])}",
                         fontsize=7.5, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row][0].set_ylabel(f"Jarak {distance_value}\n"
                                f"{exact_unique_df.loc[int(pair['idx_a']), 'label']}",
                                fontsize=9.5)

    fig.suptitle("Contoh Pasangan Duplikat Serupa pada Tiap Tingkat Jarak Hamming",
                 fontsize=12)
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/22f48c51ee6357b9e53e44f7f3c87b16604de318.png)
:::
:::

::: {#5f2790de .cell .markdown papermill="{\"duration\":3.8381e-2,\"end_time\":\"2026-08-25T22:04:38.291652+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:38.253271+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.1.2.3 Sampel Duplikat Serupa Beda Kelas {#4123-sampel-duplikat-serupa-beda-kelas}
:::

::: {#0ae67aa5 .cell .code execution_count="13" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:38.362197Z\",\"iopub.status.busy\":\"2026-08-25T22:04:38.361422Z\",\"iopub.status.idle\":\"2026-08-25T22:04:39.466641Z\",\"shell.execute_reply\":\"2026-08-25T22:04:39.465745Z\"}" papermill="{\"duration\":1.163973,\"end_time\":\"2026-08-25T22:04:39.489347+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:38.325374+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
if len(cross_class_pairs) == 0:
    print("Tidak ditemukan pasangan serupa yang berbeda kelas.")
else:
    # Pasangan beda kelas tidak dihapus, namun ditampilkan sebagai bahan
    # penilaian terhadap kemungkinan derau label
    distance_levels = [value for value in range(0, PHASH_SCAN_LIMIT + 1, 2)
                       if (cross_class_pairs["distance"] == value).any()]

    fig, axes = plt.subplots(len(distance_levels), 2,
                             figsize=(6.8, 2.8 * len(distance_levels)))
    axes = np.atleast_2d(axes)

    for row, distance_value in enumerate(distance_levels):
        pair = (cross_class_pairs[cross_class_pairs["distance"] == distance_value]
                .iloc[0])

        for col, index in enumerate([int(pair["idx_a"]), int(pair["idx_b"])]):
            member = exact_unique_df.loc[index]
            ax = axes[row][col]
            with Image.open(member["path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{member['label']}\n"
                         f"Sumber {member['source']}", fontsize=8, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row][0].set_ylabel(f"Jarak {distance_value}", fontsize=9.5)

    fig.suptitle("Contoh Pasangan Duplikat Serupa yang Berbeda Kelas", fontsize=12)
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/e7d5497285a851422964cee4c8e7d32451462955.png)
:::
:::

::: {#0a9a9814 .cell .markdown papermill="{\"duration\":5.5347e-2,\"end_time\":\"2026-08-25T22:04:39.599427+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:39.54408+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.1.2.4 Penghapusan Duplikat Serupa {#4124-penghapusan-duplikat-serupa}
:::

::: {#4ba71426 .cell .code execution_count="14" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:39.699808Z\",\"iopub.status.busy\":\"2026-08-25T22:04:39.69922Z\",\"iopub.status.idle\":\"2026-08-25T22:04:39.727569Z\",\"shell.execute_reply\":\"2026-08-25T22:04:39.726693Z\"}" papermill="{\"duration\":8.1053e-2,\"end_time\":\"2026-08-25T22:04:39.72939+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:39.648337+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
PHASH_REMOVAL = 6

removal_pairs = same_class_pairs[same_class_pairs["distance"] <= PHASH_REMOVAL]

# Union-find menggabungkan salinan yang terhubung secara tidak langsung
# sehingga setiap rantai kemiripan hanya menyisakan satu wakil
parent = list(range(len(exact_unique_df)))


def find(node):
    """Cari wakil kelompok sekaligus meratakan jalur penelusuran."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def union(node_a, node_b):
    """Gabungkan dua kelompok dengan wakil dari indeks terkecil."""
    root_a, root_b = find(node_a), find(node_b)
    if root_a != root_b:
        parent[max(root_a, root_b)] = min(root_a, root_b)


for idx_a, idx_b in zip(removal_pairs["idx_a"], removal_pairs["idx_b"]):
    union(int(idx_a), int(idx_b))

exact_unique_df["dup_group"] = [find(i) for i in range(len(exact_unique_df))]

near_group_sizes = exact_unique_df["dup_group"].value_counts()
multi_member_groups = near_group_sizes[near_group_sizes > 1]

before_near = exact_unique_df["label"].value_counts()

clean_df = (exact_unique_df
            .sort_values(["source", "filename"])
            .drop_duplicates(subset="dup_group", keep="first")
            .sort_index()
            .reset_index(drop=True))

n_before_near = len(exact_unique_df)
n_after_near = len(clean_df)
n_removed_near = n_before_near - n_after_near

after_near = clean_df["label"].value_counts()

W = 78
print("=" * W)
print("  PENGHAPUSAN DUPLIKAT SERUPA")
print("=" * W)
print(f"  {'Ambang penghapusan':<34}: <= {PHASH_REMOVAL}")
print(f"  {'Pasangan yang dipakai':<34}: {len(removal_pairs)}")
print(f"  {'Kelompok duplikat terbentuk':<34}: {len(multi_member_groups)}")
print(f"  {'Anggota terbanyak per kelompok':<34}: "
      f"{int(multi_member_groups.max()) if len(multi_member_groups) else 0}")
print("-" * W)
print(f"  {'Citra sebelum':<34}: {n_before_near} (100.00%)")
print(f"  {'Citra dihapus':<34}: {n_removed_near} "
      f"({n_removed_near / n_before_near * 100:.2f}%)")
print(f"  {'Citra tersisa':<34}: {n_after_near} "
      f"({n_after_near / n_before_near * 100:.2f}%)")
print("-" * W)
print("  Komposisi Kelas")
print(f"    {'Kelas':<26}{'Sebelum':>7}{'Dihapus':>10}{'Sesudah':>10}")
for class_name in CLASS_NAMES:
    n_class_before = int(before_near.get(class_name, 0))
    n_class_after = int(after_near.get(class_name, 0))
    print(f"    {class_name:<26}{n_class_before:>7}"
          f"{n_class_before - n_class_after:>10}{n_class_after:>10}")
print("-" * W)
print(f"  Rasio {CLASS_NAMES[0]} : {CLASS_NAMES[1]} = "
      f"{format_class_ratio(after_near)} "
      f"(sebelumnya {format_class_ratio(before_near)})")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PENGHAPUSAN DUPLIKAT SERUPA
    ==============================================================================
      Ambang penghapusan                : <= 6
      Pasangan yang dipakai             : 1097
      Kelompok duplikat terbentuk       : 964
      Anggota terbanyak per kelompok    : 6
    ------------------------------------------------------------------------------
      Citra sebelum                     : 4296 (100.00%)
      Citra dihapus                     : 1022 (23.79%)
      Citra tersisa                     : 3274 (76.21%)
    ------------------------------------------------------------------------------
      Komposisi Kelas
        Kelas                     Sebelum   Dihapus   Sesudah
        dermatitis                   1973        53      1920
        dermatophytosis              2323       969      1354
    ------------------------------------------------------------------------------
      Rasio dermatitis : dermatophytosis = 1.42 : 1.00 (sebelumnya 0.85 : 1.00)
    ==============================================================================
:::
:::

::: {#0d4166d5 .cell .markdown papermill="{\"duration\":5.0784e-2,\"end_time\":\"2026-08-25T22:04:39.829461+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:39.778677+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.1.3 Ringkasan Hasil Pembersihan Duplikat {#413-ringkasan-hasil-pembersihan-duplikat}
:::

::: {#cdc9a249 .cell .code execution_count="15" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:39.929646Z\",\"iopub.status.busy\":\"2026-08-25T22:04:39.92891Z\",\"iopub.status.idle\":\"2026-08-25T22:04:39.938224Z\",\"shell.execute_reply\":\"2026-08-25T22:04:39.9373Z\"}" papermill="{\"duration\":6.0759e-2,\"end_time\":\"2026-08-25T22:04:39.939744+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:39.878985+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
initial_counts = image_df["label"].value_counts()

stages = [
    ("Citra utuh awal", initial_counts),
    ("Setelah duplikat identik", after_counts),
    ("Setelah duplikat serupa", after_near),
]

n_removed_total = n_intact - n_after_near

W = 78
print("=" * W)
print("  RINGKASAN PEMBERSIHAN DUPLIKAT")
print("=" * W)
print(f"    {'Tahap':<28}{CLASS_NAMES[0]:>12}{CLASS_NAMES[1]:>18}"
      f"{'Rasio':>16}")
for stage_name, counts in stages:
    print(f"    {stage_name:<28}"
          f"{int(counts.get(CLASS_NAMES[0], 0)):>12}"
          f"{int(counts.get(CLASS_NAMES[1], 0)):>18}"
          f"{format_class_ratio(counts):>16}")
print("-" * W)
print(f"  {'Total citra dihapus':<34}: {n_removed_total} "
      f"({n_removed_total / n_intact * 100:.2f}%)")
print(f"  {'Total citra tersisa':<34}: {n_after_near} "
      f"({n_after_near / n_intact * 100:.2f}%)")
print("-" * W)
print("  Porsi Citra yang Bertahan")
for class_name in CLASS_NAMES:
    n_class_initial = int(initial_counts.get(class_name, 0))
    n_class_final = int(after_near.get(class_name, 0))
    print(f"  {'  ' + class_name:<34}: "
          f"{n_class_final / n_class_initial * 100:.2f}%")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      RINGKASAN PEMBERSIHAN DUPLIKAT
    ==============================================================================
        Tahap                         dermatitis   dermatophytosis           Rasio
        Citra utuh awal                     5490              4497     1.22 : 1.00
        Setelah duplikat identik            1973              2323     0.85 : 1.00
        Setelah duplikat serupa             1920              1354     1.42 : 1.00
    ------------------------------------------------------------------------------
      Total citra dihapus               : 6713 (67.22%)
      Total citra tersisa               : 3274 (32.78%)
    ------------------------------------------------------------------------------
      Porsi Citra yang Bertahan
        dermatitis                      : 34.97%
        dermatophytosis                 : 30.11%
    ==============================================================================
:::
:::

::: {#af6fbb7c .cell .markdown papermill="{\"duration\":4.996e-2,\"end_time\":\"2026-08-25T22:04:40.042298+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:39.992338+00:00\",\"status\":\"completed\"}" tags="[]"}
## 4.2 Penghapusan Citra Drug Eruption {#42-penghapusan-citra-drug-eruption}
:::

::: {#7b3392f3 .cell .markdown papermill="{\"duration\":5.0057e-2,\"end_time\":\"2026-08-25T22:04:40.140829+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:40.090772+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.2.1 Pemeriksaan Citra Drug Eruption {#421-pemeriksaan-citra-drug-eruption}
:::

::: {#626089c7 .cell .code execution_count="16" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:40.24043Z\",\"iopub.status.busy\":\"2026-08-25T22:04:40.23961Z\",\"iopub.status.idle\":\"2026-08-25T22:04:40.251384Z\",\"shell.execute_reply\":\"2026-08-25T22:04:40.250408Z\"}" papermill="{\"duration\":6.3111e-2,\"end_time\":\"2026-08-25T22:04:40.253+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:40.189889+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
DRUG_PATTERN = "drug"

drug_mask = clean_df["filename"].str.lower().str.contains(DRUG_PATTERN)
drug_df = clean_df[drug_mask]
drug_labels = drug_df["label"].value_counts()

n_drug = len(drug_df)

W = 78
print("=" * W)
print("  PEMERIKSAAN CITRA DRUG ERUPTION")
print("=" * W)
print(f"  {'Pola pencarian nama berkas':<34}: '{DRUG_PATTERN}'")
print(f"  {'Citra diperiksa':<34}: {len(clean_df)}")
print(f"  {'Citra terdeteksi':<34}: {n_drug} "
      f"({n_drug / len(clean_df) * 100:.2f}%)")
print("-" * W)
print("  Label Citra Terdeteksi")
for class_name in CLASS_NAMES:
    print(f"  {'  ' + class_name:<34}: "
          f"{int(drug_labels.get(class_name, 0))}")
print("-" * W)
print("  Asal Sumber")
for source_code, count in drug_df["source"].value_counts().sort_index().items():
    print(f"  {'  ' + source_code + '. ' + SOURCE_MAP[source_code]:<34}: {count}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PEMERIKSAAN CITRA DRUG ERUPTION
    ==============================================================================
      Pola pencarian nama berkas        : 'drug'
      Citra diperiksa                   : 3274
      Citra terdeteksi                  : 225 (6.87%)
    ------------------------------------------------------------------------------
      Label Citra Terdeteksi
        dermatitis                      : 224
        dermatophytosis                 : 1
    ------------------------------------------------------------------------------
      Asal Sumber
        A. DermNet                      : 2
        B. Skin Disease Image Dataset   : 212
        E. Skin Diseases Cancer Comprehensive Dataset: 10
        G. Tinea Clean DermNet Roboflow : 1
    ==============================================================================
:::
:::

::: {#23d113e4 .cell .markdown papermill="{\"duration\":4.9192e-2,\"end_time\":\"2026-08-25T22:04:40.352958+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:40.303766+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.2.2 Sampel Citra Drug Eruption {#422-sampel-citra-drug-eruption}
:::

::: {#feb1daf4 .cell .code execution_count="17" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:40.45244Z\",\"iopub.status.busy\":\"2026-08-25T22:04:40.451712Z\",\"iopub.status.idle\":\"2026-08-25T22:04:41.167246Z\",\"shell.execute_reply\":\"2026-08-25T22:04:41.166423Z\"}" papermill="{\"duration\":0.778105,\"end_time\":\"2026-08-25T22:04:41.179823+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:40.401718+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
N_SHOW = 8
N_COLS = 4

if n_drug == 0:
    print("Tidak ada citra drug eruption untuk ditampilkan.")
else:
    # Contoh diambil merata sepanjang subset agar tidak terpusat pada satu sumber
    positions = np.linspace(0, n_drug - 1, min(N_SHOW, n_drug)).astype(int)
    samples = drug_df.iloc[positions]

    n_rows = int(np.ceil(len(samples) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(3.2 * N_COLS, 2.7 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, record) in zip(axes, samples.iterrows()):
        with Image.open(record["path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"Sumber {record['source']}\n"
                     f"{shorten_filename(record['filename'])}",
                     fontsize=8, pad=4)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.set_visible(False)

    fig.suptitle("Contoh Citra Drug Eruption, seluruhnya berlabel "
                 f"{CLASS_NAMES[0]}", fontsize=12)
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/c6fc4f61aa9e2dad455ec14eb06866cc808c86ac.png)
:::
:::

::: {#49616cfc .cell .markdown papermill="{\"duration\":6.1628e-2,\"end_time\":\"2026-08-25T22:04:41.305067+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:41.243439+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.2.3 Penghapusan Citra Drug Eruption {#423-penghapusan-citra-drug-eruption}
:::

::: {#837e89da .cell .code execution_count="18" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:41.421841Z\",\"iopub.status.busy\":\"2026-08-25T22:04:41.420925Z\",\"iopub.status.idle\":\"2026-08-25T22:04:41.435413Z\",\"shell.execute_reply\":\"2026-08-25T22:04:41.434516Z\"}" papermill="{\"duration\":7.4977e-2,\"end_time\":\"2026-08-25T22:04:41.437047+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:41.36207+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
before_drug = clean_df["label"].value_counts()
n_before_drug = len(clean_df)

clean_df = clean_df[~drug_mask].reset_index(drop=True)

n_clean = len(clean_df)
n_removed_drug = n_before_drug - n_clean
after_drug = clean_df["label"].value_counts()

W = 78
print("=" * W)
print("  PENGHAPUSAN CITRA DRUG ERUPTION")
print("=" * W)
print(f"  {'Citra sebelum':<34}: {n_before_drug} (100.00%)")
print(f"  {'Citra dihapus':<34}: {n_removed_drug} "
      f"({n_removed_drug / n_before_drug * 100:.2f}%)")
print(f"  {'Citra tersisa':<34}: {n_clean} "
      f"({n_clean / n_before_drug * 100:.2f}%)")
print("-" * W)
print("  Komposisi Kelas")
print(f"    {'Kelas':<26}{'Sebelum':>7}{'Dihapus':>10}{'Sesudah':>10}")
for class_name in CLASS_NAMES:
    n_class_before = int(before_drug.get(class_name, 0))
    n_class_after = int(after_drug.get(class_name, 0))
    print(f"    {class_name:<26}{n_class_before:>7}"
          f"{n_class_before - n_class_after:>10}{n_class_after:>10}")
print("-" * W)
print(f"  Rasio {CLASS_NAMES[0]} : {CLASS_NAMES[1]} = "
      f"{format_class_ratio(after_drug)} "
      f"(sebelumnya {format_class_ratio(before_drug)})")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PENGHAPUSAN CITRA DRUG ERUPTION
    ==============================================================================
      Citra sebelum                     : 3274 (100.00%)
      Citra dihapus                     : 225 (6.87%)
      Citra tersisa                     : 3049 (93.13%)
    ------------------------------------------------------------------------------
      Komposisi Kelas
        Kelas                     Sebelum   Dihapus   Sesudah
        dermatitis                   1920       224      1696
        dermatophytosis              1354         1      1353
    ------------------------------------------------------------------------------
      Rasio dermatitis : dermatophytosis = 1.25 : 1.00 (sebelumnya 1.42 : 1.00)
    ==============================================================================
:::
:::

::: {#644e6bfc .cell .markdown papermill="{\"duration\":5.7143e-2,\"end_time\":\"2026-08-25T22:04:41.55111+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:41.493967+00:00\",\"status\":\"completed\"}" tags="[]"}
## 4.3 Transformasi Citra {#43-transformasi-citra}
:::

::: {#49d03aba .cell .markdown papermill="{\"duration\":5.7322e-2,\"end_time\":\"2026-08-25T22:04:41.667001+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:41.609679+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.3.1 Kanonisasi Orientasi {#431-kanonisasi-orientasi}
:::

::: {#7018e31f .cell .markdown papermill="{\"duration\":5.8257e-2,\"end_time\":\"2026-08-25T22:04:41.781717+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:41.72346+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.3.1.1 Penerapan Kanonisasi Orientasi {#4311-penerapan-kanonisasi-orientasi}
:::

::: {#7fb98992 .cell .code execution_count="19" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:41.900733Z\",\"iopub.status.busy\":\"2026-08-25T22:04:41.899949Z\",\"iopub.status.idle\":\"2026-08-25T22:04:41.913712Z\",\"shell.execute_reply\":\"2026-08-25T22:04:41.912645Z\"}" papermill="{\"duration\":7.6524e-2,\"end_time\":\"2026-08-25T22:04:41.915626+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:41.839102+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def canonical_orientation(img):
    """Putar citra tegak menjadi mendatar tanpa membuang piksel."""
    width, height = img.size
    if height > width:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


# Keputusan kanonisasi dicatat sekarang, sedangkan pemutaran piksel
# dijalankan saat citra dimuat pada tahap 4.4
clean_df["rotated"] = clean_df["orientation"] == "portrait"
clean_df["orientation_final"] = np.where(clean_df["rotated"], "landscape",
                                         clean_df["orientation"])

orientation_before = clean_df["orientation"].value_counts()
orientation_after = clean_df["orientation_final"].value_counts()

n_clean = len(clean_df)
n_rotated = int(clean_df["rotated"].sum())

W = 78
print("=" * W)
print("  KANONISASI ORIENTASI")
print("=" * W)
print(f"  {'Aturan':<34}: citra portrait diputar 90 derajat")
print(f"  {'Piksel yang hilang':<34}: tidak ada")
print("-" * W)
print("  Sebaran Orientasi")
print(f"    {'Orientasi':<26}{'Sebelum':>10}{'Sesudah':>10}")
for orientation_name in ORIENTATIONS:
    print(f"    {orientation_name:<26}"
          f"{int(orientation_before.get(orientation_name, 0)):>10}"
          f"{int(orientation_after.get(orientation_name, 0)):>10}")
print("-" * W)
print(f"  {'Citra diputar':<34}: {n_rotated} "
      f"({n_rotated / n_clean * 100:.2f}%)")
print(f"  {'Citra tidak diputar':<34}: {n_clean - n_rotated} "
      f"({(n_clean - n_rotated) / n_clean * 100:.2f}%)")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      KANONISASI ORIENTASI
    ==============================================================================
      Aturan                            : citra portrait diputar 90 derajat
      Piksel yang hilang                : tidak ada
    ------------------------------------------------------------------------------
      Sebaran Orientasi
        Orientasi                    Sebelum   Sesudah
        landscape                       2204      3039
        portrait                         835         0
        persegi                           10        10
    ------------------------------------------------------------------------------
      Citra diputar                     : 835 (27.39%)
      Citra tidak diputar               : 2214 (72.61%)
    ==============================================================================
:::
:::

::: {#08bf7223 .cell .markdown papermill="{\"duration\":7.1426e-2,\"end_time\":\"2026-08-25T22:04:42.050734+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:41.979308+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.3.1.2 Sampel Hasil Kanonisasi Orientasi {#4312-sampel-hasil-kanonisasi-orientasi}
:::

::: {#d7c5a2a8 .cell .code execution_count="20" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:42.200317Z\",\"iopub.status.busy\":\"2026-08-25T22:04:42.199415Z\",\"iopub.status.idle\":\"2026-08-25T22:04:42.797864Z\",\"shell.execute_reply\":\"2026-08-25T22:04:42.797102Z\"}" papermill="{\"duration\":0.686773,\"end_time\":\"2026-08-25T22:04:42.812967+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:42.126194+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
SAMPLE_COUNT = 3

rotated_df = clean_df[clean_df["rotated"]]

if len(rotated_df) == 0:
    print("Tidak ada citra yang memerlukan kanonisasi orientasi.")
else:
    # Contoh diambil merata sepanjang subset agar tidak terpusat pada satu sumber
    positions = np.linspace(0, len(rotated_df) - 1, SAMPLE_COUNT).astype(int)
    samples = rotated_df.iloc[positions]

    fig, axes = plt.subplots(SAMPLE_COUNT, 2,
                             figsize=(7.0, 3.1 * SAMPLE_COUNT))
    axes = np.atleast_2d(axes)

    for row, (_, record) in enumerate(samples.iterrows()):
        with Image.open(record["path"]) as img:
            original = img.convert("RGB")
        canonical = canonical_orientation(original)

        for col, (image_obj, caption) in enumerate([(original, "Sebelum"),
                                                    (canonical, "Sesudah")]):
            ax = axes[row][col]
            ax.imshow(image_obj)
            ax.set_title(f"{caption}\n"
                         f"{image_obj.size[0]} x {image_obj.size[1]} piksel",
                         fontsize=8, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row][0].set_ylabel(f"Sumber {record['source']}\n{record['label']}",
                                fontsize=9)

    fig.suptitle("Contoh Hasil Kanonisasi Orientasi", fontsize=12)
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/33f67389a4168f8f403ba5a572e0d496391156df.png)
:::
:::

::: {#74e99e9a .cell .markdown papermill="{\"duration\":8.2373e-2,\"end_time\":\"2026-08-25T22:04:42.971107+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:42.888734+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.3.2 Pemberian Padding (Letterboxing) {#432-pemberian-padding-letterboxing}
:::

::: {#3984f4cf .cell .markdown papermill="{\"duration\":6.9642e-2,\"end_time\":\"2026-08-25T22:04:43.108781+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:43.039139+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.3.2.1 Penerapan Letterboxing {#4321-penerapan-letterboxing}
:::

::: {#4adc48b3 .cell .code execution_count="21" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:43.272828Z\",\"iopub.status.busy\":\"2026-08-25T22:04:43.271836Z\",\"iopub.status.idle\":\"2026-08-25T22:04:43.284744Z\",\"shell.execute_reply\":\"2026-08-25T22:04:43.283773Z\"}" papermill="{\"duration\":8.3143e-2,\"end_time\":\"2026-08-25T22:04:43.286164+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:43.203021+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
PADDING_COLOR = (0, 0, 0)


def letterbox_resize(img, target_size=IMAGE_SIZE, fill_color=PADDING_COLOR):
    """Perkecil citra hingga muat pada kanvas, sisi kosong diisi bantalan."""
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
    """Kanonisasi orientasi lalu letterbox ke kanvas target."""
    return letterbox_resize(canonical_orientation(img), target_size)


# Porsi kanvas yang menjadi bantalan bergantung pada selisih rasio citra
# terhadap rasio kanvas, dihitung setelah orientasi dikanonisasi
clean_aspect = clean_df["aspect"].to_numpy()
ratio_gap = (np.maximum(clean_aspect, canvas_ratio)
             / np.minimum(clean_aspect, canvas_ratio))
padding_share = 1 - 1 / ratio_gap

n_no_padding = int((padding_share < 0.01).sum())

W = 78
print("=" * W)
print("  LETTERBOXING")
print("=" * W)
print(f"  {'Kanvas target':<34}: {IMAGE_WIDTH} x {IMAGE_HEIGHT} piksel")
print(f"  {'Rasio kanvas':<34}: {canvas_ratio:.2f} : 1")
print(f"  {'Warna bantalan':<34}: hitam {PADDING_COLOR}")
print(f"  {'Metode penskalaan':<34}: bilinear")
print("-" * W)
print("  Porsi Kanvas yang Menjadi Bantalan")
print(f"  {'  Rata-rata':<34}: {padding_share.mean() * 100:.2f}%")
print(f"  {'  Median':<34}: {np.median(padding_share) * 100:.2f}%")
print(f"  {'  Persentil 90':<34}: {np.percentile(padding_share, 90) * 100:.2f}%")
print(f"  {'  Maksimum':<34}: {padding_share.max() * 100:.2f}%")
print("-" * W)
print(f"  {'Citra tanpa bantalan berarti':<34}: {n_no_padding} "
      f"({n_no_padding / n_clean * 100:.2f}%)")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      LETTERBOXING
    ==============================================================================
      Kanvas target                     : 336 x 224 piksel
      Rasio kanvas                      : 1.50 : 1
      Warna bantalan                    : hitam (0, 0, 0)
      Metode penskalaan                 : bilinear
    ------------------------------------------------------------------------------
      Porsi Kanvas yang Menjadi Bantalan
        Rata-rata                       : 3.38%
        Median                          : 1.67%
        Persentil 90                    : 11.50%
        Maksimum                        : 50.00%
    ------------------------------------------------------------------------------
      Citra tanpa bantalan berarti      : 1289 (42.28%)
    ==============================================================================
:::
:::

::: {#a1048562 .cell .markdown papermill="{\"duration\":6.5857e-2,\"end_time\":\"2026-08-25T22:04:43.420141+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:43.354284+00:00\",\"status\":\"completed\"}" tags="[]"}
#### 4.3.2.2 Sampel Hasil Letterboxing {#4322-sampel-hasil-letterboxing}
:::

::: {#80b320d0 .cell .code execution_count="22" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:43.555438Z\",\"iopub.status.busy\":\"2026-08-25T22:04:43.554618Z\",\"iopub.status.idle\":\"2026-08-25T22:04:44.132292Z\",\"shell.execute_reply\":\"2026-08-25T22:04:44.131566Z\"}" papermill="{\"duration\":0.656735,\"end_time\":\"2026-08-25T22:04:44.144818+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:43.488083+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
# Contoh dipilih menurut besar bantalan agar rentang dampak letterboxing
# terlihat, dari yang hampir tidak berbantalan hingga yang terbesar
padding_rank = np.argsort(padding_share)
sample_positions = [
    ("Bantalan terkecil", padding_rank[0]),
    ("Bantalan menengah", padding_rank[len(padding_rank) // 2]),
    ("Bantalan terbesar", padding_rank[-1]),
]

fig, axes = plt.subplots(len(sample_positions), 2,
                         figsize=(7.6, 2.9 * len(sample_positions)))
axes = np.atleast_2d(axes)

for row, (caption, position) in enumerate(sample_positions):
    record = clean_df.iloc[position]

    with Image.open(record["path"]) as img:
        original = img.convert("RGB")
    transformed = transform_image(original)

    for col, (image_obj, panel_title) in enumerate([
        (original, f"Sebelum\n{original.size[0]} x {original.size[1]} piksel"),
        (transformed, f"Sesudah\n{IMAGE_WIDTH} x {IMAGE_HEIGHT} piksel"),
    ]):
        ax = axes[row][col]
        ax.imshow(image_obj)
        ax.set_title(panel_title, fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[row][0].set_ylabel(f"{caption}\n"
                            f"{padding_share[position] * 100:.2f}% kanvas",
                            fontsize=9)

fig.suptitle("Contoh Hasil Letterboxing pada Berbagai Besar Bantalan", fontsize=12)
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/835c4d6a6ae073b1cb73a28fabb02cdb239f4ba9.png)
:::
:::

::: {#857c5631 .cell .markdown papermill="{\"duration\":7.765e-2,\"end_time\":\"2026-08-25T22:04:44.304762+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:44.227112+00:00\",\"status\":\"completed\"}" tags="[]"}
## 4.4 Pembagian Data {#44-pembagian-data}
:::

::: {#cb4319dc .cell .markdown papermill="{\"duration\":7.4909e-2,\"end_time\":\"2026-08-25T22:04:44.452721+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:44.377812+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.4.1 Pembagian Stratified Data Split {#441-pembagian-stratified-data-split}
:::

::: {#73de7569 .cell .code execution_count="23" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:44.602715Z\",\"iopub.status.busy\":\"2026-08-25T22:04:44.601888Z\",\"iopub.status.idle\":\"2026-08-25T22:04:44.624701Z\",\"shell.execute_reply\":\"2026-08-25T22:04:44.623821Z\"}" papermill="{\"duration\":9.9884e-2,\"end_time\":\"2026-08-25T22:04:44.6262+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:44.526316+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "validation", "test")
train_ratio, val_ratio, test_ratio = SPLIT_RATIOS

# Pembagian dilakukan atas citra tunggal dan distratifikasi per kelas agar
# proporsi kedua kelas seragam pada ketiga bagian.
train_val_df, test_df = train_test_split(
    clean_df, test_size=test_ratio, random_state=SEED,
    stratify=clean_df["label"])

# Porsi validasi dihitung ulang relatif terhadap sisa data setelah data uji
# dipisahkan, agar proporsi akhirnya sesuai target
relative_val = val_ratio / (train_ratio + val_ratio)
train_df, val_df = train_test_split(
    train_val_df, test_size=relative_val, random_state=SEED,
    stratify=train_val_df["label"])

# Penanda bagian dikembalikan ke clean_df sebelum indeks disusun ulang,
# untuk keperluan verifikasi
clean_df["split"] = "train"
clean_df.loc[val_df.index, "split"] = "validation"
clean_df.loc[test_df.index, "split"] = "test"

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

n_split_total = len(clean_df)

W = 78
print("=" * W)
print("  PEMBAGIAN DATA")
print("=" * W)
print(f"  {'Proporsi target':<34}: "
      f"{' : '.join(str(int(ratio * 100)) for ratio in SPLIT_RATIOS)}")
print(f"  {'Satuan pembagian':<34}: citra tunggal")
print(f"  {'Stratifikasi':<34}: per kelas")
print(f"  {'Seed pengacakan':<34}: {SEED}")
print("-" * W)
print(f"  {'Bagian':<16}{'Citra':>8}{'Porsi':>10}{'Target':>10}")
for split_name, ratio, part in zip(SPLIT_NAMES, SPLIT_RATIOS,
                                   [train_df, val_df, test_df]):
    print(f"  {split_name:<16}{len(part):>8}"
          f"{len(part) / n_split_total * 100:>9.2f}%{ratio * 100:>9.1f}%")
print("-" * W)
print(f"  {'Total':<16}{n_split_total:>8}{'100.00%':>10}{'100.0%':>10}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PEMBAGIAN DATA
    ==============================================================================
      Proporsi target                   : 70 : 15 : 15
      Satuan pembagian                  : citra tunggal
      Stratifikasi                      : per kelas
      Seed pengacakan                   : 42
    ------------------------------------------------------------------------------
      Bagian             Citra     Porsi    Target
      train               2133    69.96%     70.0%
      validation           458    15.02%     15.0%
      test                 458    15.02%     15.0%
    ------------------------------------------------------------------------------
      Total               3049   100.00%    100.0%
    ==============================================================================
:::
:::

::: {#b1e0db33 .cell .markdown papermill="{\"duration\":7.5779e-2,\"end_time\":\"2026-08-25T22:04:44.777366+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:44.701587+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.4.2 Verifikasi Hasil Pembagian {#442-verifikasi-hasil-pembagian}
:::

::: {#80052c4f .cell .code execution_count="24" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:44.934659Z\",\"iopub.status.busy\":\"2026-08-25T22:04:44.934192Z\",\"iopub.status.idle\":\"2026-08-25T22:04:44.944993Z\",\"shell.execute_reply\":\"2026-08-25T22:04:44.944243Z\"}" papermill="{\"duration\":9.3511e-2,\"end_time\":\"2026-08-25T22:04:44.946854+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:44.853343+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
n_counted = len(train_df) + len(val_df) + len(test_df)
n_unassigned = int(clean_df["split"].isna().sum())
overall_counts = clean_df["label"].value_counts()

W = 78
print("=" * W)
print("  VERIFIKASI HASIL PEMBAGIAN")
print("=" * W)
print(f"  {'Citra tanpa bagian':<34}: {n_unassigned}")
print(f"  {'Selisih jumlah citra':<34}: {len(clean_df) - n_counted}")
print("-" * W)
print("  Komposisi Kelas per Bagian")
print(f"    {'Bagian':<16}{CLASS_NAMES[0]:>12}{CLASS_NAMES[1]:>18}{'Rasio':>16}")
for split_name, part in zip(SPLIT_NAMES, [train_df, val_df, test_df]):
    counts = part["label"].value_counts()
    print(f"    {split_name:<16}"
          f"{int(counts.get(CLASS_NAMES[0], 0)):>12}"
          f"{int(counts.get(CLASS_NAMES[1], 0)):>18}"
          f"{format_class_ratio(counts):>16}")
print("-" * W)
print(f"    {'Seluruh data':<16}"
      f"{int(overall_counts.get(CLASS_NAMES[0], 0)):>12}"
      f"{int(overall_counts.get(CLASS_NAMES[1], 0)):>18}"
      f"{format_class_ratio(overall_counts):>16}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      VERIFIKASI HASIL PEMBAGIAN
    ==============================================================================
      Citra tanpa bagian                : 0
      Selisih jumlah citra              : 0
    ------------------------------------------------------------------------------
      Komposisi Kelas per Bagian
        Bagian            dermatitis   dermatophytosis           Rasio
        train                   1186               947     1.25 : 1.00
        validation               255               203     1.26 : 1.00
        test                     255               203     1.26 : 1.00
    ------------------------------------------------------------------------------
        Seluruh data            1696              1353     1.25 : 1.00
    ==============================================================================
:::
:::

::: {#0a7bd0f3 .cell .markdown papermill="{\"duration\":7.9443e-2,\"end_time\":\"2026-08-25T22:04:45.101164+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:45.021721+00:00\",\"status\":\"completed\"}" tags="[]"}
## 4.5 Pemuatan Data ke Memori {#45-pemuatan-data-ke-memori}
:::

::: {#34110df7 .cell .code execution_count="25" execution="{\"iopub.execute_input\":\"2026-08-25T22:04:45.260291Z\",\"iopub.status.busy\":\"2026-08-25T22:04:45.259826Z\",\"iopub.status.idle\":\"2026-08-25T22:05:04.198981Z\",\"shell.execute_reply\":\"2026-08-25T22:05:04.197985Z\"}" papermill="{\"duration\":19.021511,\"end_time\":\"2026-08-25T22:05:04.200844+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:04:45.179333+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def preload_split(split_part):
    """Muat seluruh citra satu bagian ke array uint8 beserta label binernya."""
    images = np.empty((len(split_part), IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                      dtype=np.uint8)
    labels = np.empty(len(split_part), dtype=np.float32)

    for index, record in enumerate(split_part.itertuples(index=False)):
        with Image.open(record.path) as img:
            images[index] = np.asarray(transform_image(img.convert("RGB")),
                                       dtype=np.uint8)
        labels[index] = float(CLASS_NAMES.index(record.label))

    return images, labels


def array_size_mb(array):
    """Ukuran array dalam megabita."""
    return array.nbytes / (1024 ** 2)


X_train, y_train = preload_split(train_df)
X_val, y_val = preload_split(val_df)
X_test, y_test = preload_split(test_df)

split_arrays = list(zip(SPLIT_NAMES,
                        [X_train, X_val, X_test],
                        [y_train, y_val, y_test]))
total_mb = sum(array_size_mb(images) for _, images, _ in split_arrays)

W = 78
print("=" * W)
print("  PEMUATAN DATA KE MEMORI")
print("=" * W)
print(f"  {'Bentuk citra':<34}: {IMAGE_HEIGHT} x {IMAGE_WIDTH} x 3")
print(f"  {'Tipe data citra':<34}: {X_train.dtype}")
print(f"  {'Label kelas 0 / 1':<34}: {CLASS_NAMES[0]} / {CLASS_NAMES[1]}")
print("-" * W)
print(f"    {'Bagian':<14}{'Bentuk array':<24}{'Kelas 0':>9}{'Kelas 1':>9}"
      f"{'Memori':>12}")
for split_name, images, labels in split_arrays:
    print(f"    {split_name:<14}{str(images.shape):<24}"
          f"{int((labels == 0).sum()):>9}{int((labels == 1).sum()):>9}"
          f"{array_size_mb(images):>9.1f} MB")
print("-" * W)
print(f"  {'Total memori citra':<34}: {total_mb:.1f} MB")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PEMUATAN DATA KE MEMORI
    ==============================================================================
      Bentuk citra                      : 224 x 336 x 3
      Tipe data citra                   : uint8
      Label kelas 0 / 1                 : dermatitis / dermatophytosis
    ------------------------------------------------------------------------------
        Bagian        Bentuk array              Kelas 0  Kelas 1      Memori
        train         (2133, 224, 336, 3)          1186      947    459.3 MB
        validation    (458, 224, 336, 3)            255      203     98.6 MB
        test          (458, 224, 336, 3)            255      203     98.6 MB
    ------------------------------------------------------------------------------
      Total memori citra                : 656.5 MB
    ==============================================================================
:::
:::

::: {#9b5d893f .cell .markdown papermill="{\"duration\":7.9284e-2,\"end_time\":\"2026-08-25T22:05:04.363951+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:04.284667+00:00\",\"status\":\"completed\"}" tags="[]"}
## 4.6 Normalisasi Nilai Piksel {#46-normalisasi-nilai-piksel}
:::

::: {#93a8c25f .cell .code execution_count="26" execution="{\"iopub.execute_input\":\"2026-08-25T22:05:04.518271Z\",\"iopub.status.busy\":\"2026-08-25T22:05:04.517387Z\",\"iopub.status.idle\":\"2026-08-25T22:05:04.587896Z\",\"shell.execute_reply\":\"2026-08-25T22:05:04.586969Z\"}" papermill="{\"duration\":0.150096,\"end_time\":\"2026-08-25T22:05:04.589803+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:04.439707+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def normalize_img(image, label):
    """Ubah rentang nilai piksel dari 0-255 menjadi 0-1."""
    return tf.cast(image, tf.float32) / 255.0, label


# Verifikasi dilakukan pada satu batch agar tidak menyalin seluruh data
# latih ke dalam bentuk pecahan
sample_images = X_train[:BATCH_SIZE]
normalized_images, _ = normalize_img(sample_images, y_train[:BATCH_SIZE])
normalized_images = normalized_images.numpy()

W = 78
print("=" * W)
print("  NORMALISASI NILAI PIKSEL")
print("=" * W)
print(f"  {'Operasi':<34}: nilai piksel dibagi 255.0")
print(f"  {'Perubahan tipe data':<34}: {sample_images.dtype} -> "
      f"{normalized_images.dtype}")
print(f"  {'Waktu penerapan':<34}: per batch, sebelum augmentasi")
print("-" * W)
print(f"  Verifikasi pada {len(sample_images)} citra pertama data latih")
print(f"  {'  Rentang sebelum':<34}: {sample_images.min()} - "
      f"{sample_images.max()}")
print(f"  {'  Rentang sesudah':<34}: {normalized_images.min():.4f} - "
      f"{normalized_images.max():.4f}")
print(f"  {'  Rata-rata sesudah':<34}: {normalized_images.mean():.4f}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      NORMALISASI NILAI PIKSEL
    ==============================================================================
      Operasi                           : nilai piksel dibagi 255.0
      Perubahan tipe data               : uint8 -> float32
      Waktu penerapan                   : per batch, sebelum augmentasi
    ------------------------------------------------------------------------------
      Verifikasi pada 32 citra pertama data latih
        Rentang sebelum                 : 0 - 255
        Rentang sesudah                 : 0.0000 - 1.0000
        Rata-rata sesudah               : 0.3649
    ==============================================================================
:::
:::

::: {#80b3ffb3 .cell .markdown papermill="{\"duration\":7.7507e-2,\"end_time\":\"2026-08-25T22:05:04.74445+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:04.666943+00:00\",\"status\":\"completed\"}" tags="[]"}
## 4.7 Augmentasi Citra {#47-augmentasi-citra}
:::

::: {#654355d9 .cell .markdown papermill="{\"duration\":7.7413e-2,\"end_time\":\"2026-08-25T22:05:04.898025+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:04.820612+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.7.1 Konfigurasi Augmentasi {#471-konfigurasi-augmentasi}
:::

::: {#1d81573c .cell .code execution_count="27" execution="{\"iopub.execute_input\":\"2026-08-25T22:05:05.052862Z\",\"iopub.status.busy\":\"2026-08-25T22:05:05.052407Z\",\"iopub.status.idle\":\"2026-08-25T22:05:05.061137Z\",\"shell.execute_reply\":\"2026-08-25T22:05:05.06024Z\"}" papermill="{\"duration\":8.8734e-2,\"end_time\":\"2026-08-25T22:05:05.062726+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:04.973992+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
ZOOM_MIN = 0.7
BRIGHTNESS_MAX = 0.2


def augment_img(image, label):
    """Terapkan transformasi acak yang tidak mengubah kelas citra.

    Seluruh transformasi bekerja pada citra masukan dan tidak menyentuh
    label. Hanya terjadi satu kali penyampelan ulang, yaitu saat citra
    dikembalikan ke ukuran kanvas setelah pemotongan.
    """
    # Empat simetri persegi panjang dicapai lewat dua pembalikan bebas
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Perbesaran dilakukan dengan memotong acak lalu mengembalikan ukuran
    zoom_scale = tf.random.uniform([], ZOOM_MIN, 1.0)
    crop_height = tf.cast(tf.cast(IMAGE_HEIGHT, tf.float32) * zoom_scale, tf.int32)
    crop_width = tf.cast(tf.cast(IMAGE_WIDTH, tf.float32) * zoom_scale, tf.int32)
    image = tf.image.random_crop(image, size=[crop_height, crop_width, 3])
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])

    # Perubahan kecerahan, hasilnya dijaga tetap pada rentang 0-1
    image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_MAX)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


W = 78
print("=" * W)
print("  KONFIGURASI AUGMENTASI")
print("=" * W)
print(f"  {'Cakupan penerapan':<34}: data latih saja")
print(f"  {'Waktu penerapan':<34}: per batch, setelah normalisasi")
print(f"  {'Pengaruh terhadap label':<34}: tidak ada")
print("-" * W)
print("  Transformasi")
print(f"  {'  Pembalikan horizontal':<34}: acak, peluang 0.5")
print(f"  {'  Pembalikan vertikal':<34}: acak, peluang 0.5")
print(f"  {'  Perbesaran':<34}: {ZOOM_MIN} - 1.0 "
      f"(luas tersisa {ZOOM_MIN ** 2 * 100:.0f}% - 100%)")
print(f"  {'  Kecerahan':<34}: +/- {BRIGHTNESS_MAX}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      KONFIGURASI AUGMENTASI
    ==============================================================================
      Cakupan penerapan                 : data latih saja
      Waktu penerapan                   : per batch, setelah normalisasi
      Pengaruh terhadap label           : tidak ada
    ------------------------------------------------------------------------------
      Transformasi
        Pembalikan horizontal           : acak, peluang 0.5
        Pembalikan vertikal             : acak, peluang 0.5
        Perbesaran                      : 0.7 - 1.0 (luas tersisa 49% - 100%)
        Kecerahan                       : +/- 0.2
    ==============================================================================
:::
:::

::: {#6a3c1743 .cell .markdown papermill="{\"duration\":7.6448e-2,\"end_time\":\"2026-08-25T22:05:05.214865+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:05.138417+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.7.2 Sampel Hasil Augmentasi {#472-sampel-hasil-augmentasi}
:::

::: {#4ea93ac9 .cell .code execution_count="28" execution="{\"iopub.execute_input\":\"2026-08-25T22:05:05.368371Z\",\"iopub.status.busy\":\"2026-08-25T22:05:05.367944Z\",\"iopub.status.idle\":\"2026-08-25T22:05:08.214364Z\",\"shell.execute_reply\":\"2026-08-25T22:05:08.213451Z\"}" papermill="{\"duration\":2.942538,\"end_time\":\"2026-08-25T22:05:08.232883+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:05.290345+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
N_AUGMENTED = 7
N_COLS = 4


def show_augmentation(image_uint8, label_value, class_name):
    """Tampilkan citra asli beserta beberapa hasil augmentasinya."""
    samples = []
    for _ in range(N_AUGMENTED):
        normalized, _ = normalize_img(image_uint8, label_value)
        augmented, _ = augment_img(normalized, label_value)
        samples.append(augmented.numpy())

    n_rows = int(np.ceil((N_AUGMENTED + 1) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(3.2 * N_COLS, 2.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(image_uint8)
    axes[0].set_title("Asli", fontsize=9, pad=4)

    for index, augmented in enumerate(samples, start=1):
        axes[index].imshow(augmented)
        axes[index].set_title(f"Augmentasi {index}", fontsize=9, pad=4)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Hasil Augmentasi Kelas {class_name}", fontsize=12)
    plt.tight_layout()
    plt.show()


# Satu citra pertama dari tiap kelas dipakai sebagai contoh
for class_index, class_name in enumerate(CLASS_NAMES):
    position = int(np.argmax(y_train == class_index))
    show_augmentation(X_train[position], y_train[position], class_name)
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/4f661fd4777769e29d0589d6a8ed85eb3366357a.png)
:::

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/1e761810e7792d34552da7ef5c9861ee63381e55.png)
:::
:::

::: {#5caee356 .cell .markdown papermill="{\"duration\":0.116401,\"end_time\":\"2026-08-25T22:05:08.468615+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:08.352214+00:00\",\"status\":\"completed\"}" tags="[]"}
### 4.7.3 Penyusunan Pipeline Data {#473-penyusunan-pipeline-data}
:::

::: {#eb8b05a3 .cell .code execution_count="29" execution="{\"iopub.execute_input\":\"2026-08-25T22:05:08.682887Z\",\"iopub.status.busy\":\"2026-08-25T22:05:08.682409Z\",\"iopub.status.idle\":\"2026-08-25T22:05:13.491848Z\",\"shell.execute_reply\":\"2026-08-25T22:05:13.491131Z\"}" papermill="{\"duration\":4.922318,\"end_time\":\"2026-08-25T22:05:13.493465+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:08.571147+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
AUTOTUNE = tf.data.AUTOTUNE

# Pipeline latih, satu-satunya yang dikocok dan diaugmentasi
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train), seed=TRAINING_SEED,
             reshuffle_each_iteration=True)
    .map(normalize_img, num_parallel_calls=AUTOTUNE)
    .map(augment_img, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# Data latih tanpa augmentasi, dipakai saat pengukuran kinerja
train_eval_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(normalize_img, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(normalize_img, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(normalize_img, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

pipelines = [
    ("train", train_ds, "ya", "ya"),
    ("train (evaluasi)", train_eval_ds, "tidak", "tidak"),
    ("validation", val_ds, "tidak", "tidak"),
    ("test", test_ds, "tidak", "tidak"),
]

sample_batch, sample_labels = next(iter(train_ds))
label_values = np.unique(sample_labels.numpy())

W = 78
print("=" * W)
print("  PIPELINE DATA")
print("=" * W)
print(f"  {'Ukuran batch':<34}: {BATCH_SIZE}")
print(f"  {'Seed pengocokan':<34}: {TRAINING_SEED}")
print(f"  {'Pengocokan ulang tiap epoch':<34}: ya")
print("-" * W)
print(f"    {'Pipeline':<20}{'Augmentasi':>12}{'Pengocokan':>13}{'Batch':>9}")
for name, pipeline, augmented, shuffled in pipelines:
    print(f"    {name:<20}{augmented:>12}{shuffled:>13}{len(pipeline):>9}")
print("-" * W)
print("  Pemeriksaan Satu Batch Latih")
print(f"  {'  Bentuk citra':<34}: {tuple(sample_batch.shape)}")
print(f"  {'  Tipe data':<34}: {sample_batch.dtype.name}")
print(f"  {'  Rentang nilai':<34}: "
      f"{float(tf.reduce_min(sample_batch)):.2f} - "
      f"{float(tf.reduce_max(sample_batch)):.2f}")
print(f"  {'  Nilai label yang muncul':<34}: "
      f"{', '.join(f'{value:.0f}' for value in label_values)}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PIPELINE DATA
    ==============================================================================
      Ukuran batch                      : 32
      Seed pengocokan                   : 44
      Pengocokan ulang tiap epoch       : ya
    ------------------------------------------------------------------------------
        Pipeline              Augmentasi   Pengocokan    Batch
        train                         ya           ya       67
        train (evaluasi)           tidak        tidak       67
        validation                 tidak        tidak       15
        test                       tidak        tidak       15
    ------------------------------------------------------------------------------
      Pemeriksaan Satu Batch Latih
        Bentuk citra                    : (32, 224, 336, 3)
        Tipe data                       : float32
        Rentang nilai                   : 0.00 - 1.00
        Nilai label yang muncul         : 0, 1
    ==============================================================================
:::
:::

::: {#0139f326 .cell .markdown papermill="{\"duration\":0.107861,\"end_time\":\"2026-08-25T22:05:13.707003+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:13.599142+00:00\",\"status\":\"completed\"}" tags="[]"}
# 5. Modeling {#5-modeling}
:::

::: {#34cc4816 .cell .markdown papermill="{\"duration\":0.102288,\"end_time\":\"2026-08-25T22:05:13.911291+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:13.809003+00:00\",\"status\":\"completed\"}" tags="[]"}
## 5.1 Arsitektur Model {#51-arsitektur-model}
:::

::: {#eebc92fe .cell .code execution_count="30" execution="{\"iopub.execute_input\":\"2026-08-25T22:05:14.118363Z\",\"iopub.status.busy\":\"2026-08-25T22:05:14.117723Z\",\"iopub.status.idle\":\"2026-08-25T22:05:15.102454Z\",\"shell.execute_reply\":\"2026-08-25T22:05:15.101648Z\"}" papermill="{\"duration\":1.090613,\"end_time\":\"2026-08-25T22:05:15.103995+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:14.013382+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
DROPOUT_RATE = 0.3         # lapisan padat pada kepala klasifikasi
CONV_DROPOUT_RATE = 0.15   # SpatialDropout2D pada badan konvolusi


def build_model(input_shape, num_classes=1,
                dropout_rate=DROPOUT_RATE,
                conv_dropout_rate=CONV_DROPOUT_RATE):
    """Bangun arsitektur CNN residual untuk klasifikasi biner."""
    inputs = tf.keras.Input(shape=input_shape)

    # Blok 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok 3 dan 4, disatukan oleh sambungan residual
    residual = layers.Conv2D(128, (1, 1), padding="same")(x)
    residual = layers.MaxPooling2D((2, 2))(residual)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)

    x = layers.SpatialDropout2D(conv_dropout_rate)(x)

    # Blok 5 dan 6, disatukan oleh sambungan residual
    residual = layers.Conv2D(256, (1, 1), padding="same")(x)
    residual = layers.MaxPooling2D((2, 2))(residual)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.SpatialDropout2D(conv_dropout_rate)(x)

    x = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)

    # Kepala klasifikasi
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs,
                          name="Residual_CNN_Dermatitis_Dermatophytosis")


preview_model = build_model(input_shape=INPUT_SHAPE)
preview_model.summary()

# Peta fitur terakhir dibaca dari keluaran lapisan tepat sebelum GAP
gap_index = [isinstance(layer, layers.GlobalAveragePooling2D)
             for layer in preview_model.layers].index(True)
feature_map = preview_model.layers[gap_index - 1].output.shape

head_params = sum(layer.count_params()
                  for layer in preview_model.layers[gap_index:])
total_params = preview_model.count_params()

W = 78
print()
print("=" * W)
print("  RINGKASAN ARSITEKTUR")
print("=" * W)
print(f"  {'Bentuk masukan':<34}: {INPUT_SHAPE}")
print(f"  {'Bentuk keluaran':<34}: sigmoid, 1 nilai")
print(f"  {'Peta fitur sebelum GAP':<34}: "
      f"{feature_map[1]} x {feature_map[2]} x {feature_map[3]}")
print("-" * W)
print(f"  {'Total parameter':<34}: {total_params:,}")
print(f"  {'  Badan konvolusi':<34}: {total_params - head_params:,}")
print(f"  {'  Kepala klasifikasi':<34}: {head_params:,}")
print("-" * W)
print(f"  {'Dropout kepala klasifikasi':<34}: {DROPOUT_RATE}")
print(f"  {'Dropout badan konvolusi':<34}: {CONV_DROPOUT_RATE}")
print("=" * W)

del preview_model
```

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "Residual_CNN_Dermatitis_Dermatophytosis"</span>
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">336</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">336</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">896</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalization │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">336</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">128</span> │ conv2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">168</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">168</span>,  │     <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │ max_pooling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">168</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_1     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">84</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">84</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │ max_pooling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">84</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_3     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ max_pooling2d_3[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">84</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">8,320</span> │ max_pooling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │ max_pooling2d_2[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ spatial_dropout2d   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SpatialDropout2D</span>)  │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │    <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │ spatial_dropout2… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_5     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ spatial_dropout2d_1 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ max_pooling2d_5[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SpatialDropout2D</span>)  │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ separable_conv2d    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">68,096</span> │ spatial_dropout2… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SeparableConv2D</span>)   │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">42</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">33,024</span> │ spatial_dropout2… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ separable_conv2d… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_4     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │ max_pooling2d_4[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_1        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePool…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │     <span style="color: #00af00; text-decoration-color: #00af00">65,792</span> │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dense[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │     <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │ dropout[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dense_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │        <span style="color: #00af00; text-decoration-color: #00af00">129</span> │ dropout_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">747,713</span> (2.85 MB)
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">745,985</span> (2.85 MB)
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,728</span> (6.75 KB)
</pre>
```
:::

::: {.output .stream .stdout}

    ==============================================================================
      RINGKASAN ARSITEKTUR
    ==============================================================================
      Bentuk masukan                    : (224, 336, 3)
      Bentuk keluaran                   : sigmoid, 1 nilai
      Peta fitur sebelum GAP            : 14 x 21 x 256
    ------------------------------------------------------------------------------
      Total parameter                   : 747,713
        Badan konvolusi                 : 648,896
        Kepala klasifikasi              : 98,817
    ------------------------------------------------------------------------------
      Dropout kepala klasifikasi        : 0.3
      Dropout badan konvolusi           : 0.15
    ==============================================================================
:::
:::

::: {#181e1061 .cell .markdown papermill="{\"duration\":0.10361,\"end_time\":\"2026-08-25T22:05:15.311379+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:15.207769+00:00\",\"status\":\"completed\"}" tags="[]"}
## 5.2 Konfigurasi Pelatihan {#52-konfigurasi-pelatihan}
:::

::: {#66d53764 .cell .code execution_count="31" execution="{\"iopub.execute_input\":\"2026-08-25T22:05:15.526651Z\",\"iopub.status.busy\":\"2026-08-25T22:05:15.526112Z\",\"iopub.status.idle\":\"2026-08-25T22:05:15.539717Z\",\"shell.execute_reply\":\"2026-08-25T22:05:15.538675Z\"}" papermill="{\"duration\":0.126196,\"end_time\":\"2026-08-25T22:05:15.541609+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:15.415413+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
WARMUP_EPOCHS = 10
INITIAL_LR = 1e-5
TARGET_LR = 5e-4
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1e-4

# Kriteria pemilihan bobot dan ambang klasifikasi
WEIGHT_CRITERION = "val_auc"
CRITERION_MODE = "max"
THRESHOLD = 0.5

# Parameter penormalan dan bias tidak dikenai peluruhan bobot
DECAY_EXCLUDED_NAMES = ["gamma", "beta", "bias"]

CHECKPOINT_PATH = WORKING_DIR / "ckpt.weights.h5"

steps_per_epoch = len(train_ds)
warmup_steps = WARMUP_EPOCHS * steps_per_epoch
total_steps = EPOCHS * steps_per_epoch
decay_steps = total_steps - warmup_steps


def cosine_lr_at(epoch):
    """Hitung laju pembelajaran pada awal suatu epoch."""
    step = epoch * steps_per_epoch
    if step < warmup_steps:
        return INITIAL_LR + (TARGET_LR - INITIAL_LR) * (step / warmup_steps)
    progress = min((step - warmup_steps) / decay_steps, 1.0)
    return TARGET_LR * 0.5 * (1.0 + np.cos(np.pi * progress))


class ProgressPrinter(keras.callbacks.Callback):
    """Cetak metrik latih dan validasi berdampingan pada tiap epoch."""

    def on_epoch_end(self, epoch, logs=None):
        print(f"    Epoch {epoch + 1:>3}/{EPOCHS}"
              f"   lr {cosine_lr_at(epoch + 1):.6f}"
              f"   loss {logs['loss']:.4f}"
              f"   acc {logs['accuracy'] * 100:>6.2f}%"
              f"   auc {logs['auc'] * 100:>6.2f}%"
              f"   |"
              f"   val_loss {logs['val_loss']:.4f}"
              f"   val_acc {logs['val_accuracy'] * 100:>6.2f}%"
              f"   val_auc {logs['val_auc'] * 100:>6.2f}%")


def build_compiled_model():
    """Bangun model beserta konfigurasi kompilasinya di dalam lingkup strategi."""
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=decay_steps,
        alpha=0.0,
        warmup_target=TARGET_LR,
        warmup_steps=warmup_steps
    )

    with strategy.scope():
        compiled_model = build_model(input_shape=INPUT_SHAPE)

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY
        )
        optimizer.exclude_from_weight_decay(var_names=DECAY_EXCLUDED_NAMES)

        compiled_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(
                label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

    return compiled_model


def build_callbacks():
    """Susun callback pelatihan."""
    return [
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor=WEIGHT_CRITERION,
            mode=CRITERION_MODE,
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        ),
        ProgressPrinter()
    ]


W = 78
print("=" * W)
print("  KONFIGURASI PELATIHAN")
print("=" * W)
print(f"  {'Optimizer':<34}: AdamW")
print(f"  {'Jadwal learning rate':<34}: cosine decay dengan warmup")
print(f"  {'Learning rate':<34}: {INITIAL_LR} -> {TARGET_LR} -> 0")
print(f"  {'Panjang warmup':<34}: {WARMUP_EPOCHS} epoch")
print(f"  {'Epoch':<34}: {EPOCHS}")
print(f"  {'Langkah per epoch':<34}: {steps_per_epoch}")
print("-" * W)
print(f"  {'Fungsi kerugian':<34}: binary crossentropy")
print(f"  {'Label smoothing':<34}: {LABEL_SMOOTHING}")
print(f"  {'Weight decay':<34}: {WEIGHT_DECAY}")
print(f"  {'Pengecualian weight decay':<34}: "
      f"{', '.join(DECAY_EXCLUDED_NAMES)}")
print(f"  {'Metrik pantauan':<34}: accuracy, AUC")
print("-" * W)
print(f"  {'Kriteria pemilihan bobot':<34}: {WEIGHT_CRITERION} ({CRITERION_MODE})")
print(f"  {'Ambang klasifikasi':<34}: {THRESHOLD}")
print(f"  {'Berkas checkpoint':<34}: {CHECKPOINT_PATH.name}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      KONFIGURASI PELATIHAN
    ==============================================================================
      Optimizer                         : AdamW
      Jadwal learning rate              : cosine decay dengan warmup
      Learning rate                     : 1e-05 -> 0.0005 -> 0
      Panjang warmup                    : 10 epoch
      Epoch                             : 150
      Langkah per epoch                 : 67
    ------------------------------------------------------------------------------
      Fungsi kerugian                   : binary crossentropy
      Label smoothing                   : 0.05
      Weight decay                      : 0.0001
      Pengecualian weight decay         : gamma, beta, bias
      Metrik pantauan                   : accuracy, AUC
    ------------------------------------------------------------------------------
      Kriteria pemilihan bobot          : val_auc (max)
      Ambang klasifikasi                : 0.5
      Berkas checkpoint                 : ckpt.weights.h5
    ==============================================================================
:::
:::

::: {#88e90b73 .cell .markdown papermill="{\"duration\":0.109426,\"end_time\":\"2026-08-25T22:05:15.754421+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:15.644995+00:00\",\"status\":\"completed\"}" tags="[]"}
## 5.3 Pelatihan Model {#53-pelatihan-model}
:::

::: {#e1d46be2 .cell .code execution_count="32" execution="{\"iopub.execute_input\":\"2026-08-25T22:05:15.965834Z\",\"iopub.status.busy\":\"2026-08-25T22:05:15.964884Z\",\"iopub.status.idle\":\"2026-08-25T22:31:34.936823Z\",\"shell.execute_reply\":\"2026-08-25T22:31:34.935854Z\"}" papermill="{\"duration\":1579.08048,\"end_time\":\"2026-08-25T22:31:34.938644+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:05:15.858164+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
# Seed pelatihan disetel ulang tepat sebelum model dibangun agar bobot awal
# dan urutan pengocokan dapat direproduksi
random.seed(TRAINING_SEED)
np.random.seed(TRAINING_SEED)
tf.random.set_seed(TRAINING_SEED)

final_model = build_compiled_model()

W = 78
print("=" * W)
print("  PELATIHAN MODEL")
print("=" * W)

start_time = time.time()
history = final_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=build_callbacks(),
    verbose=0
)
train_duration = time.time() - start_time

history_dict = history.history
best_epoch = int(np.argmax(history_dict[WEIGHT_CRITERION])) + 1
best_criterion_value = float(np.max(history_dict[WEIGHT_CRITERION]))

print("-" * W)
print(f"  {'Epoch dijalankan':<34}: {len(history_dict['loss'])} / {EPOCHS}")
print(f"  {'Durasi pelatihan':<34}: {train_duration / 60:.1f} menit")
print(f"  {'Rata-rata per epoch':<34}: "
      f"{train_duration / len(history_dict['loss']):.1f} detik")
print("-" * W)
print(f"  {'Bobot terbaik':<34}: epoch {best_epoch}")
print(f"  {'Nilai ' + WEIGHT_CRITERION + ' terbaik':<34}: "
      f"{best_criterion_value * 100:.2f}%")
print(f"  {'Berkas checkpoint':<34}: {CHECKPOINT_PATH.name}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PELATIHAN MODEL
    ==============================================================================
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
:::

::: {.output .stream .stderr}
    E0000 00:00:1787695520.309809      23 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/Residual_CNN_Dermatitis_Dermatophytosis_1/spatial_dropout2d_2_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
:::

::: {.output .stream .stdout}
        Epoch   1/150   lr 0.000059   loss 0.7845   acc  53.49%   auc  49.86%   |   val_loss 0.6924   val_acc  55.68%   val_auc  55.45%
        Epoch   2/150   lr 0.000108   loss 0.6994   acc  57.10%   auc  57.15%   |   val_loss 0.6905   val_acc  55.68%   val_auc  53.40%
        Epoch   3/150   lr 0.000157   loss 0.6701   acc  60.43%   auc  63.05%   |   val_loss 0.6879   val_acc  55.68%   val_auc  57.84%
        Epoch   4/150   lr 0.000206   loss 0.6596   acc  61.28%   auc  65.17%   |   val_loss 0.6830   val_acc  56.99%   val_auc  68.02%
        Epoch   5/150   lr 0.000255   loss 0.6524   acc  63.67%   auc  66.71%   |   val_loss 0.6746   val_acc  56.99%   val_auc  66.41%
        Epoch   6/150   lr 0.000304   loss 0.6497   acc  63.99%   auc  67.21%   |   val_loss 0.6670   val_acc  59.17%   val_auc  68.39%
        Epoch   7/150   lr 0.000353   loss 0.6427   acc  65.26%   auc  67.97%   |   val_loss 0.6229   val_acc  66.16%   val_auc  80.32%
        Epoch   8/150   lr 0.000402   loss 0.6453   acc  64.70%   auc  68.41%   |   val_loss 0.5978   val_acc  70.09%   val_auc  80.75%
        Epoch   9/150   lr 0.000451   loss 0.6324   acc  65.07%   auc  70.33%   |   val_loss 0.6074   val_acc  70.31%   val_auc  75.84%
        Epoch  10/150   lr 0.000500   loss 0.6406   acc  64.42%   auc  69.03%   |   val_loss 0.6080   val_acc  70.31%   val_auc  73.07%
        Epoch  11/150   lr 0.000500   loss 0.6304   acc  65.96%   auc  69.94%   |   val_loss 0.5630   val_acc  73.36%   val_auc  79.90%
        Epoch  12/150   lr 0.000500   loss 0.6257   acc  65.68%   auc  70.43%   |   val_loss 0.5979   val_acc  73.36%   val_auc  76.29%
        Epoch  13/150   lr 0.000499   loss 0.6167   acc  66.34%   auc  72.14%   |   val_loss 0.5615   val_acc  72.05%   val_auc  80.75%
        Epoch  14/150   lr 0.000499   loss 0.6273   acc  66.24%   auc  70.85%   |   val_loss 0.6095   val_acc  69.65%   val_auc  73.21%
        Epoch  15/150   lr 0.000498   loss 0.6186   acc  66.85%   auc  72.05%   |   val_loss 0.5689   val_acc  71.40%   val_auc  79.62%
        Epoch  16/150   lr 0.000498   loss 0.6193   acc  66.71%   auc  72.08%   |   val_loss 0.5846   val_acc  72.05%   val_auc  81.36%
        Epoch  17/150   lr 0.000497   loss 0.6179   acc  66.24%   auc  71.84%   |   val_loss 0.6273   val_acc  68.12%   val_auc  81.19%
        Epoch  18/150   lr 0.000496   loss 0.6079   acc  67.60%   auc  73.81%   |   val_loss 0.6682   val_acc  63.54%   val_auc  80.10%
        Epoch  19/150   lr 0.000495   loss 0.6104   acc  68.54%   auc  73.73%   |   val_loss 0.5528   val_acc  73.14%   val_auc  83.03%
        Epoch  20/150   lr 0.000494   loss 0.6062   acc  68.54%   auc  74.00%   |   val_loss 0.5934   val_acc  69.65%   val_auc  79.62%
        Epoch  21/150   lr 0.000492   loss 0.6010   acc  68.26%   auc  74.57%   |   val_loss 1.1520   val_acc  44.54%   val_auc  78.03%
        Epoch  22/150   lr 0.000491   loss 0.6046   acc  68.35%   auc  74.06%   |   val_loss 0.5538   val_acc  74.89%   val_auc  83.00%
        Epoch  23/150   lr 0.000489   loss 0.6012   acc  70.00%   auc  74.66%   |   val_loss 0.6056   val_acc  72.27%   val_auc  82.20%
        Epoch  24/150   lr 0.000488   loss 0.5971   acc  68.68%   auc  74.95%   |   val_loss 0.5276   val_acc  75.55%   val_auc  83.49%
        Epoch  25/150   lr 0.000486   loss 0.6086   acc  68.31%   auc  73.69%   |   val_loss 0.5255   val_acc  74.67%   val_auc  82.93%
        Epoch  26/150   lr 0.000484   loss 0.5988   acc  69.95%   auc  75.11%   |   val_loss 0.6297   val_acc  71.62%   val_auc  83.09%
        Epoch  27/150   lr 0.000482   loss 0.5897   acc  69.57%   auc  75.83%   |   val_loss 0.5384   val_acc  74.45%   val_auc  81.62%
        Epoch  28/150   lr 0.000480   loss 0.5891   acc  70.28%   auc  75.95%   |   val_loss 0.6482   val_acc  69.87%   val_auc  82.38%
        Epoch  29/150   lr 0.000478   loss 0.5886   acc  68.73%   auc  76.11%   |   val_loss 0.6170   val_acc  70.31%   val_auc  81.67%
        Epoch  30/150   lr 0.000475   loss 0.5886   acc  70.89%   auc  76.24%   |   val_loss 0.5406   val_acc  74.67%   val_auc  83.08%
        Epoch  31/150   lr 0.000473   loss 0.5805   acc  70.56%   auc  77.18%   |   val_loss 0.7644   val_acc  58.52%   val_auc  80.92%
        Epoch  32/150   lr 0.000470   loss 0.5769   acc  70.51%   auc  77.45%   |   val_loss 0.5676   val_acc  73.14%   val_auc  83.90%
        Epoch  33/150   lr 0.000467   loss 0.5774   acc  71.59%   auc  77.71%   |   val_loss 0.5984   val_acc  70.74%   val_auc  83.82%
        Epoch  34/150   lr 0.000465   loss 0.5774   acc  72.57%   auc  77.42%   |   val_loss 0.7078   val_acc  58.08%   val_auc  81.77%
        Epoch  35/150   lr 0.000462   loss 0.5901   acc  69.57%   auc  75.85%   |   val_loss 0.5321   val_acc  75.55%   val_auc  83.83%
        Epoch  36/150   lr 0.000459   loss 0.5725   acc  72.20%   auc  78.26%   |   val_loss 0.5135   val_acc  77.07%   val_auc  84.96%
        Epoch  37/150   lr 0.000456   loss 0.5742   acc  71.50%   auc  78.07%   |   val_loss 0.5231   val_acc  77.07%   val_auc  85.77%
        Epoch  38/150   lr 0.000452   loss 0.5695   acc  70.56%   auc  77.90%   |   val_loss 0.6473   val_acc  68.12%   val_auc  82.81%
        Epoch  39/150   lr 0.000449   loss 0.5833   acc  70.04%   auc  76.53%   |   val_loss 0.5662   val_acc  72.71%   val_auc  84.21%
        Epoch  40/150   lr 0.000445   loss 0.5603   acc  72.81%   auc  79.19%   |   val_loss 0.5171   val_acc  79.04%   val_auc  85.15%
        Epoch  41/150   lr 0.000442   loss 0.5661   acc  72.01%   auc  78.89%   |   val_loss 0.5451   val_acc  74.89%   val_auc  82.84%
        Epoch  42/150   lr 0.000438   loss 0.5724   acc  72.29%   auc  78.29%   |   val_loss 0.5102   val_acc  76.20%   val_auc  84.93%
        Epoch  43/150   lr 0.000435   loss 0.5650   acc  72.53%   auc  78.96%   |   val_loss 0.5487   val_acc  75.33%   val_auc  84.05%
        Epoch  44/150   lr 0.000431   loss 0.5635   acc  71.50%   auc  78.83%   |   val_loss 0.6214   val_acc  75.55%   val_auc  76.89%
        Epoch  45/150   lr 0.000427   loss 0.5569   acc  73.00%   auc  79.55%   |   val_loss 0.6354   val_acc  70.09%   val_auc  83.62%
        Epoch  46/150   lr 0.000423   loss 0.5624   acc  71.92%   auc  78.91%   |   val_loss 0.5412   val_acc  74.45%   val_auc  81.88%
        Epoch  47/150   lr 0.000419   loss 0.5613   acc  71.64%   auc  78.99%   |   val_loss 0.5722   val_acc  73.36%   val_auc  84.73%
        Epoch  48/150   lr 0.000414   loss 0.5509   acc  73.84%   auc  80.20%   |   val_loss 0.5064   val_acc  77.95%   val_auc  85.49%
        Epoch  49/150   lr 0.000410   loss 0.5429   acc  73.93%   auc  80.91%   |   val_loss 0.5063   val_acc  77.51%   val_auc  85.11%
        Epoch  50/150   lr 0.000406   loss 0.5499   acc  73.93%   auc  80.90%   |   val_loss 0.6100   val_acc  70.52%   val_auc  77.43%
        Epoch  51/150   lr 0.000401   loss 0.5487   acc  73.61%   auc  80.38%   |   val_loss 0.5836   val_acc  73.36%   val_auc  83.66%
        Epoch  52/150   lr 0.000397   loss 0.5385   acc  74.92%   auc  81.30%   |   val_loss 0.5584   val_acc  75.55%   val_auc  83.66%
        Epoch  53/150   lr 0.000392   loss 0.5475   acc  72.48%   auc  80.45%   |   val_loss 0.5364   val_acc  75.76%   val_auc  84.82%
        Epoch  54/150   lr 0.000388   loss 0.5449   acc  73.93%   auc  80.93%   |   val_loss 0.5108   val_acc  76.42%   val_auc  85.87%
        Epoch  55/150   lr 0.000383   loss 0.5397   acc  74.78%   auc  81.26%   |   val_loss 0.5467   val_acc  72.93%   val_auc  84.03%
        Epoch  56/150   lr 0.000378   loss 0.5321   acc  73.98%   auc  81.86%   |   val_loss 0.5837   val_acc  74.02%   val_auc  85.34%
        Epoch  57/150   lr 0.000373   loss 0.5357   acc  74.73%   auc  81.73%   |   val_loss 0.5903   val_acc  73.14%   val_auc  84.41%
        Epoch  58/150   lr 0.000368   loss 0.5308   acc  75.39%   auc  82.02%   |   val_loss 0.5652   val_acc  75.98%   val_auc  84.71%
        Epoch  59/150   lr 0.000363   loss 0.5306   acc  74.59%   auc  82.09%   |   val_loss 0.6319   val_acc  74.24%   val_auc  75.97%
        Epoch  60/150   lr 0.000358   loss 0.5281   acc  75.86%   auc  82.61%   |   val_loss 0.5922   val_acc  70.09%   val_auc  86.01%
        Epoch  61/150   lr 0.000353   loss 0.5200   acc  75.53%   auc  83.05%   |   val_loss 0.5777   val_acc  73.58%   val_auc  85.83%
        Epoch  62/150   lr 0.000348   loss 0.5326   acc  74.96%   auc  82.27%   |   val_loss 0.4895   val_acc  79.48%   val_auc  86.02%
        Epoch  63/150   lr 0.000343   loss 0.5285   acc  74.68%   auc  82.30%   |   val_loss 0.6038   val_acc  72.93%   val_auc  85.46%
        Epoch  64/150   lr 0.000338   loss 0.5106   acc  75.95%   auc  84.31%   |   val_loss 0.5696   val_acc  75.76%   val_auc  85.23%
        Epoch  65/150   lr 0.000333   loss 0.5144   acc  76.79%   auc  83.74%   |   val_loss 0.4879   val_acc  77.73%   val_auc  85.69%
        Epoch  66/150   lr 0.000327   loss 0.5192   acc  76.42%   auc  83.45%   |   val_loss 0.5322   val_acc  76.42%   val_auc  85.38%
        Epoch  67/150   lr 0.000322   loss 0.5065   acc  77.17%   auc  84.70%   |   val_loss 0.5029   val_acc  77.95%   val_auc  86.29%
        Epoch  68/150   lr 0.000317   loss 0.5098   acc  75.81%   auc  84.14%   |   val_loss 0.5037   val_acc  78.82%   val_auc  85.88%
        Epoch  69/150   lr 0.000311   loss 0.5108   acc  76.61%   auc  84.14%   |   val_loss 0.5746   val_acc  75.11%   val_auc  85.68%
        Epoch  70/150   lr 0.000306   loss 0.4981   acc  77.45%   auc  85.40%   |   val_loss 0.5503   val_acc  76.42%   val_auc  86.60%
        Epoch  71/150   lr 0.000300   loss 0.5144   acc  76.28%   auc  83.71%   |   val_loss 0.4787   val_acc  79.04%   val_auc  87.04%
        Epoch  72/150   lr 0.000295   loss 0.4899   acc  77.64%   auc  85.82%   |   val_loss 0.4685   val_acc  79.04%   val_auc  87.50%
        Epoch  73/150   lr 0.000289   loss 0.4972   acc  77.45%   auc  85.24%   |   val_loss 0.5803   val_acc  72.93%   val_auc  85.87%
        Epoch  74/150   lr 0.000284   loss 0.4845   acc  78.90%   auc  86.55%   |   val_loss 0.5005   val_acc  78.60%   val_auc  87.45%
        Epoch  75/150   lr 0.000278   loss 0.4831   acc  79.47%   auc  86.47%   |   val_loss 0.5928   val_acc  71.40%   val_auc  83.50%
        Epoch  76/150   lr 0.000272   loss 0.4927   acc  77.92%   auc  85.59%   |   val_loss 0.5573   val_acc  74.24%   val_auc  85.45%
        Epoch  77/150   lr 0.000267   loss 0.4848   acc  78.62%   auc  86.21%   |   val_loss 0.4900   val_acc  79.69%   val_auc  87.25%
        Epoch  78/150   lr 0.000261   loss 0.4740   acc  79.65%   auc  86.94%   |   val_loss 0.5011   val_acc  79.26%   val_auc  86.90%
        Epoch  79/150   lr 0.000256   loss 0.4904   acc  77.12%   auc  85.63%   |   val_loss 0.4905   val_acc  77.73%   val_auc  86.89%
        Epoch  80/150   lr 0.000250   loss 0.4898   acc  79.23%   auc  85.86%   |   val_loss 0.5005   val_acc  77.29%   val_auc  87.46%
        Epoch  81/150   lr 0.000244   loss 0.4793   acc  78.72%   auc  86.91%   |   val_loss 0.5015   val_acc  78.82%   val_auc  86.01%
        Epoch  82/150   lr 0.000239   loss 0.4807   acc  78.76%   auc  86.49%   |   val_loss 0.4644   val_acc  81.22%   val_auc  88.06%
        Epoch  83/150   lr 0.000233   loss 0.4710   acc  78.81%   auc  87.39%   |   val_loss 0.4918   val_acc  79.26%   val_auc  87.80%
        Epoch  84/150   lr 0.000228   loss 0.4731   acc  79.70%   auc  86.98%   |   val_loss 0.4839   val_acc  81.44%   val_auc  87.92%
        Epoch  85/150   lr 0.000222   loss 0.4696   acc  79.09%   auc  87.53%   |   val_loss 0.5156   val_acc  77.51%   val_auc  87.84%
        Epoch  86/150   lr 0.000216   loss 0.4719   acc  79.98%   auc  87.40%   |   val_loss 0.4610   val_acc  81.00%   val_auc  89.39%
        Epoch  87/150   lr 0.000211   loss 0.4653   acc  80.78%   auc  87.79%   |   val_loss 0.5353   val_acc  78.17%   val_auc  88.40%
        Epoch  88/150   lr 0.000205   loss 0.4650   acc  80.45%   auc  87.67%   |   val_loss 0.4728   val_acc  80.57%   val_auc  87.80%
        Epoch  89/150   lr 0.000200   loss 0.4655   acc  79.65%   auc  87.57%   |   val_loss 0.5332   val_acc  77.29%   val_auc  88.38%
        Epoch  90/150   lr 0.000194   loss 0.4506   acc  81.11%   auc  88.82%   |   val_loss 0.4876   val_acc  80.13%   val_auc  88.75%
        Epoch  91/150   lr 0.000189   loss 0.4520   acc  80.83%   auc  88.66%   |   val_loss 0.5256   val_acc  79.48%   val_auc  88.37%
        Epoch  92/150   lr 0.000183   loss 0.4450   acc  80.59%   auc  89.13%   |   val_loss 0.4752   val_acc  81.00%   val_auc  88.20%
        Epoch  93/150   lr 0.000178   loss 0.4535   acc  81.15%   auc  88.57%   |   val_loss 0.4503   val_acc  81.44%   val_auc  88.80%
        Epoch  94/150   lr 0.000173   loss 0.4413   acc  81.20%   auc  89.44%   |   val_loss 0.4799   val_acc  79.69%   val_auc  87.64%
        Epoch  95/150   lr 0.000167   loss 0.4420   acc  81.20%   auc  89.51%   |   val_loss 0.4864   val_acc  80.13%   val_auc  87.68%
        Epoch  96/150   lr 0.000162   loss 0.4336   acc  81.11%   auc  90.00%   |   val_loss 0.4694   val_acc  80.57%   val_auc  87.28%
        Epoch  97/150   lr 0.000157   loss 0.4292   acc  81.76%   auc  90.30%   |   val_loss 0.5070   val_acc  77.95%   val_auc  88.28%
        Epoch  98/150   lr 0.000152   loss 0.4235   acc  83.36%   auc  90.67%   |   val_loss 0.4500   val_acc  82.75%   val_auc  89.71%
        Epoch  99/150   lr 0.000147   loss 0.4371   acc  82.65%   auc  89.67%   |   val_loss 0.4768   val_acc  81.66%   val_auc  89.22%
        Epoch 100/150   lr 0.000142   loss 0.4255   acc  82.65%   auc  90.49%   |   val_loss 0.4848   val_acc  78.38%   val_auc  88.89%
        Epoch 101/150   lr 0.000137   loss 0.4143   acc  83.40%   auc  91.24%   |   val_loss 0.4622   val_acc  80.57%   val_auc  89.55%
        Epoch 102/150   lr 0.000132   loss 0.4242   acc  82.70%   auc  90.55%   |   val_loss 0.4719   val_acc  80.13%   val_auc  88.99%
        Epoch 103/150   lr 0.000127   loss 0.4241   acc  83.17%   auc  90.63%   |   val_loss 0.4685   val_acc  80.57%   val_auc  88.44%
        Epoch 104/150   lr 0.000122   loss 0.4191   acc  82.93%   auc  91.00%   |   val_loss 0.4968   val_acc  80.35%   val_auc  88.60%
        Epoch 105/150   lr 0.000117   loss 0.4205   acc  82.89%   auc  90.78%   |   val_loss 0.4710   val_acc  81.66%   val_auc  89.03%
        Epoch 106/150   lr 0.000112   loss 0.4131   acc  83.68%   auc  91.37%   |   val_loss 0.4434   val_acc  82.10%   val_auc  90.05%
        Epoch 107/150   lr 0.000108   loss 0.4006   acc  82.89%   auc  92.15%   |   val_loss 0.4590   val_acc  81.00%   val_auc  89.68%
        Epoch 108/150   lr 0.000103   loss 0.4082   acc  84.15%   auc  91.58%   |   val_loss 0.4654   val_acc  81.22%   val_auc  89.05%
        Epoch 109/150   lr 0.000099   loss 0.3998   acc  84.29%   auc  92.03%   |   val_loss 0.4395   val_acc  81.88%   val_auc  90.67%
        Epoch 110/150   lr 0.000094   loss 0.4140   acc  83.12%   auc  91.29%   |   val_loss 0.4267   val_acc  82.75%   val_auc  90.78%
        Epoch 111/150   lr 0.000090   loss 0.4080   acc  83.97%   auc  91.56%   |   val_loss 0.4448   val_acc  82.31%   val_auc  90.48%
        Epoch 112/150   lr 0.000086   loss 0.4003   acc  84.90%   auc  92.08%   |   val_loss 0.4367   val_acc  82.31%   val_auc  89.88%
        Epoch 113/150   lr 0.000081   loss 0.3987   acc  84.95%   auc  92.13%   |   val_loss 0.4428   val_acc  81.00%   val_auc  90.32%
        Epoch 114/150   lr 0.000077   loss 0.3969   acc  84.67%   auc  92.31%   |   val_loss 0.4478   val_acc  83.41%   val_auc  89.68%
        Epoch 115/150   lr 0.000073   loss 0.3917   acc  85.33%   auc  92.70%   |   val_loss 0.4422   val_acc  82.97%   val_auc  90.60%
        Epoch 116/150   lr 0.000069   loss 0.3997   acc  83.87%   auc  92.06%   |   val_loss 0.4448   val_acc  80.79%   val_auc  90.72%
        Epoch 117/150   lr 0.000065   loss 0.3940   acc  85.09%   auc  92.39%   |   val_loss 0.4579   val_acc  81.66%   val_auc  90.80%
        Epoch 118/150   lr 0.000062   loss 0.3954   acc  84.90%   auc  92.34%   |   val_loss 0.4469   val_acc  82.31%   val_auc  89.99%
        Epoch 119/150   lr 0.000058   loss 0.3927   acc  85.28%   auc  92.47%   |   val_loss 0.4568   val_acc  81.22%   val_auc  89.25%
        Epoch 120/150   lr 0.000055   loss 0.3879   acc  85.14%   auc  92.79%   |   val_loss 0.4618   val_acc  81.88%   val_auc  89.96%
        Epoch 121/150   lr 0.000051   loss 0.3884   acc  85.28%   auc  92.71%   |   val_loss 0.4656   val_acc  80.57%   val_auc  90.78%
        Epoch 122/150   lr 0.000048   loss 0.3946   acc  84.95%   auc  92.41%   |   val_loss 0.4332   val_acc  83.41%   val_auc  90.85%
        Epoch 123/150   lr 0.000044   loss 0.3825   acc  85.09%   auc  93.04%   |   val_loss 0.4405   val_acc  82.97%   val_auc  90.54%
        Epoch 124/150   lr 0.000041   loss 0.3856   acc  85.04%   auc  92.87%   |   val_loss 0.4374   val_acc  82.53%   val_auc  90.69%
        Epoch 125/150   lr 0.000038   loss 0.3818   acc  85.47%   auc  93.08%   |   val_loss 0.4428   val_acc  82.97%   val_auc  90.76%
        Epoch 126/150   lr 0.000035   loss 0.3713   acc  86.22%   auc  93.74%   |   val_loss 0.4560   val_acc  81.22%   val_auc  89.90%
        Epoch 127/150   lr 0.000033   loss 0.3818   acc  85.23%   auc  93.01%   |   val_loss 0.4353   val_acc  83.19%   val_auc  91.01%
        Epoch 128/150   lr 0.000030   loss 0.3720   acc  85.75%   auc  93.72%   |   val_loss 0.4513   val_acc  82.10%   val_auc  90.62%
        Epoch 129/150   lr 0.000027   loss 0.3894   acc  84.58%   auc  92.69%   |   val_loss 0.4430   val_acc  82.53%   val_auc  90.45%
        Epoch 130/150   lr 0.000025   loss 0.3735   acc  85.42%   auc  93.57%   |   val_loss 0.4398   val_acc  82.53%   val_auc  90.73%
        Epoch 131/150   lr 0.000022   loss 0.3652   acc  86.50%   auc  94.01%   |   val_loss 0.4373   val_acc  82.75%   val_auc  91.07%
        Epoch 132/150   lr 0.000020   loss 0.3634   acc  86.54%   auc  94.14%   |   val_loss 0.4415   val_acc  82.75%   val_auc  90.63%
        Epoch 133/150   lr 0.000018   loss 0.3730   acc  86.03%   auc  93.57%   |   val_loss 0.4397   val_acc  83.41%   val_auc  90.75%
        Epoch 134/150   lr 0.000016   loss 0.3743   acc  86.97%   auc  93.40%   |   val_loss 0.4297   val_acc  83.41%   val_auc  91.13%
        Epoch 135/150   lr 0.000014   loss 0.3751   acc  85.23%   auc  93.50%   |   val_loss 0.4340   val_acc  83.19%   val_auc  90.93%
        Epoch 136/150   lr 0.000012   loss 0.3596   acc  87.20%   auc  94.28%   |   val_loss 0.4365   val_acc  82.75%   val_auc  90.97%
        Epoch 137/150   lr 0.000011   loss 0.3689   acc  86.26%   auc  93.72%   |   val_loss 0.4367   val_acc  83.41%   val_auc  90.89%
        Epoch 138/150   lr 0.000009   loss 0.3752   acc  86.03%   auc  93.36%   |   val_loss 0.4387   val_acc  83.19%   val_auc  91.09%
        Epoch 139/150   lr 0.000008   loss 0.3616   acc  86.69%   auc  94.24%   |   val_loss 0.4412   val_acc  82.75%   val_auc  90.82%
        Epoch 140/150   lr 0.000006   loss 0.3680   acc  85.84%   auc  93.91%   |   val_loss 0.4430   val_acc  82.53%   val_auc  90.83%
        Epoch 141/150   lr 0.000005   loss 0.3694   acc  85.56%   auc  93.86%   |   val_loss 0.4395   val_acc  82.75%   val_auc  91.03%
        Epoch 142/150   lr 0.000004   loss 0.3747   acc  86.45%   auc  93.67%   |   val_loss 0.4370   val_acc  82.97%   val_auc  91.13%
        Epoch 143/150   lr 0.000003   loss 0.3712   acc  86.26%   auc  93.73%   |   val_loss 0.4360   val_acc  83.19%   val_auc  91.02%
        Epoch 144/150   lr 0.000002   loss 0.3604   acc  86.92%   auc  94.21%   |   val_loss 0.4371   val_acc  82.97%   val_auc  90.91%
        Epoch 145/150   lr 0.000002   loss 0.3566   acc  87.15%   auc  94.51%   |   val_loss 0.4363   val_acc  82.97%   val_auc  90.98%
        Epoch 146/150   lr 0.000001   loss 0.3514   acc  87.25%   auc  94.73%   |   val_loss 0.4364   val_acc  83.19%   val_auc  90.96%
        Epoch 147/150   lr 0.000001   loss 0.3625   acc  86.64%   auc  94.13%   |   val_loss 0.4360   val_acc  83.19%   val_auc  91.00%
        Epoch 148/150   lr 0.000000   loss 0.3593   acc  87.11%   auc  94.25%   |   val_loss 0.4363   val_acc  83.19%   val_auc  91.00%
        Epoch 149/150   lr 0.000000   loss 0.3686   acc  86.36%   auc  93.83%   |   val_loss 0.4365   val_acc  83.19%   val_auc  90.99%
        Epoch 150/150   lr 0.000000   loss 0.3515   acc  87.90%   auc  94.68%   |   val_loss 0.4360   val_acc  83.19%   val_auc  91.02%
    ------------------------------------------------------------------------------
      Epoch dijalankan                  : 150 / 150
      Durasi pelatihan                  : 26.3 menit
      Rata-rata per epoch               : 10.5 detik
    ------------------------------------------------------------------------------
      Bobot terbaik                     : epoch 134
      Nilai val_auc terbaik             : 91.13%
      Berkas checkpoint                 : ckpt.weights.h5
    ==============================================================================
:::
:::

::: {#9a8aa167 .cell .markdown papermill="{\"duration\":0.111502,\"end_time\":\"2026-08-25T22:31:35.164046+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:35.052544+00:00\",\"status\":\"completed\"}" tags="[]"}
## 5.4 Pemantauan Proses Pelatihan {#54-pemantauan-proses-pelatihan}
:::

::: {#b95f620b .cell .code execution_count="33" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:35.38348Z\",\"iopub.status.busy\":\"2026-08-25T22:31:35.382592Z\",\"iopub.status.idle\":\"2026-08-25T22:31:35.797131Z\",\"shell.execute_reply\":\"2026-08-25T22:31:35.79637Z\"}" papermill="{\"duration\":0.528865,\"end_time\":\"2026-08-25T22:31:35.801255+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:35.27239+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
val_loss_curve = np.array(history_dict["val_loss"])
floor_epoch = int(np.argmin(val_loss_curve)) + 1
val_loss_floor = float(val_loss_curve.min())
val_loss_tail = float(val_loss_curve[-5:].mean())

CURVE_COLOR = "#4C72B0"

panels = [
    ("loss", "val_loss", "Kurva Loss", "Loss"),
    ("accuracy", "val_accuracy", "Kurva Accuracy", "Accuracy"),
    ("auc", "val_auc", "Kurva ROC-AUC", "ROC-AUC"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
fig.suptitle("Kurva Pelatihan", fontsize=12)

for ax, (train_key, val_key, title, ylabel) in zip(axes, panels):
    ax.plot(history_dict[train_key], color=CURVE_COLOR, lw=1.3,
            label="Latih (teraugmentasi)")
    ax.plot(history_dict[val_key], color=CURVE_COLOR, lw=1.3, linestyle="--",
            label="Validasi")
    ax.axvline(best_epoch - 1, color="#C62828", lw=1.0, linestyle="--",
               alpha=0.8, label=f"Bobot terpilih (epoch {best_epoch})")
    ax.set_title(title, fontsize=10.5, fontweight="normal")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.show()

W = 78
print("=" * W)
print("  PEMANTAUAN PROSES PELATIHAN")
print("=" * W)
print(f"  {'Epoch dengan val_loss terendah':<34}: {floor_epoch}")
print(f"  {'Nilai val_loss terendah':<34}: {val_loss_floor:.4f}")
print(f"  {'Rata-rata val_loss 5 epoch akhir':<34}: {val_loss_tail:.4f}")
print(f"  {'Kenaikan rata-rata val_loss akhir':<34}: "
      f"{(val_loss_tail / val_loss_floor - 1) * 100:.2f}% dari titik terendah")
print("-" * W)
print(f"  {'Akurasi latih epoch akhir':<34}: "
      f"{history_dict['accuracy'][-1] * 100:.2f}% (teraugmentasi)")
print(f"  {'Akurasi validasi epoch akhir':<34}: "
      f"{history_dict['val_accuracy'][-1] * 100:.2f}%")
print("=" * W)
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/646fd5f76f0ca6a9f741a513a8232d55f14e7fd4.png)
:::

::: {.output .stream .stdout}
    ==============================================================================
      PEMANTAUAN PROSES PELATIHAN
    ==============================================================================
      Epoch dengan val_loss terendah    : 110
      Nilai val_loss terendah           : 0.4267
      Rata-rata val_loss 5 epoch akhir  : 0.4362
      Kenaikan rata-rata val_loss akhir : 2.24% dari titik terendah
    ------------------------------------------------------------------------------
      Akurasi latih epoch akhir         : 87.90% (teraugmentasi)
      Akurasi validasi epoch akhir      : 83.19%
    ==============================================================================
:::
:::

::: {#271e9a45 .cell .markdown papermill="{\"duration\":0.12101,\"end_time\":\"2026-08-25T22:31:36.054734+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:35.933724+00:00\",\"status\":\"completed\"}" tags="[]"}
## 5.5 Penetapan Model Final {#55-penetapan-model-final}
:::

::: {#78bdfcff .cell .code execution_count="34" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:36.311269Z\",\"iopub.status.busy\":\"2026-08-25T22:31:36.310828Z\",\"iopub.status.idle\":\"2026-08-25T22:31:40.823747Z\",\"shell.execute_reply\":\"2026-08-25T22:31:40.822791Z\"}" papermill="{\"duration\":4.64286,\"end_time\":\"2026-08-25T22:31:40.825284+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:36.182424+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
# Bobot epoch terakhir digantikan oleh bobot terbaik menurut kriteria
final_model.load_weights(CHECKPOINT_PATH)

# Data latih dinilai ulang tanpa augmentasi agar sebanding dengan validasi
train_clean = final_model.evaluate(train_eval_ds, verbose=0, return_dict=True)
val_clean = final_model.evaluate(val_ds, verbose=0, return_dict=True)

train_acc_clean = float(train_clean["accuracy"])
train_auc_clean = float(train_clean["auc"])
val_acc_clean = float(val_clean["accuracy"])
val_auc_clean = float(val_clean["auc"])

W = 78
print("=" * W)
print("  PENETAPAN MODEL FINAL")
print("=" * W)
print(f"  {'Sumber bobot':<34}: epoch {best_epoch}")
print(f"  {'Kriteria pemilihan':<34}: {WEIGHT_CRITERION} ({CRITERION_MODE})")
print(f"  {'Total parameter':<34}: {final_model.count_params():,}")
print("-" * W)
print("  Kinerja Bobot Terpilih tanpa Augmentasi")
print(f"    {'Bagian':<20}{'Loss':>12}{'Accuracy':>12}{'ROC-AUC':>12}")
for split_name, metrics in [("train", train_clean), ("validation", val_clean)]:
    print(f"    {split_name:<20}{metrics['loss']:>12.4f}"
          f"{metrics['accuracy'] * 100:>11.2f}%{metrics['auc'] * 100:>11.2f}%")
print("-" * W)
print(f"  {'Selisih akurasi latih - validasi':<34}: "
      f"{(train_acc_clean - val_acc_clean) * 100:.2f} poin")
print(f"  {'Selisih ROC-AUC latih - validasi':<34}: "
      f"{(train_auc_clean - val_auc_clean) * 100:.2f} poin")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PENETAPAN MODEL FINAL
    ==============================================================================
      Sumber bobot                      : epoch 134
      Kriteria pemilihan                : val_auc (max)
      Total parameter                   : 747,713
    ------------------------------------------------------------------------------
      Kinerja Bobot Terpilih tanpa Augmentasi
        Bagian                      Loss    Accuracy     ROC-AUC
        train                     0.3065      89.83%      96.87%
        validation                0.4297      83.41%      91.13%
    ------------------------------------------------------------------------------
      Selisih akurasi latih - validasi  : 6.42 poin
      Selisih ROC-AUC latih - validasi  : 5.73 poin
    ==============================================================================
:::
:::

::: {#0dd22b07 .cell .markdown papermill="{\"duration\":0.129089,\"end_time\":\"2026-08-25T22:31:41.075192+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:40.946103+00:00\",\"status\":\"completed\"}" tags="[]"}
# 6. Evaluation {#6-evaluation}
:::

::: {#df5b6e79 .cell .markdown papermill="{\"duration\":0.119008,\"end_time\":\"2026-08-25T22:31:41.325198+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:41.20619+00:00\",\"status\":\"completed\"}" tags="[]"}
## 6.1 Kinerja Model pada Subset Data {#61-kinerja-model-pada-subset-data}
:::

::: {#f4ccc3f4 .cell .code execution_count="35" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:41.547288Z\",\"iopub.status.busy\":\"2026-08-25T22:31:41.546292Z\",\"iopub.status.idle\":\"2026-08-25T22:31:42.475287Z\",\"shell.execute_reply\":\"2026-08-25T22:31:42.474339Z\"}" papermill="{\"duration\":1.042938,\"end_time\":\"2026-08-25T22:31:42.476794+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:41.433856+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
# Ketiga bagian diukur dengan cara yang sama, tanpa augmentasi
test_clean = final_model.evaluate(test_ds, verbose=0, return_dict=True)

split_metrics = [
    ("train", train_df, train_clean),
    ("validation", val_df, val_clean),
    ("test", test_df, test_clean),
]

test_acc_clean = float(test_clean["accuracy"])
test_auc_clean = float(test_clean["auc"])

W = 78
print("=" * W)
print("  KINERJA MODEL PADA SUBSET DATA")
print("=" * W)
print(f"  {'Sumber bobot':<34}: epoch {best_epoch}")
print(f"  {'Ambang klasifikasi':<34}: {THRESHOLD}")
print("-" * W)
print(f"    {'Bagian':<16}{'Citra':>8}{'Loss':>12}{'Accuracy':>13}"
      f"{'ROC-AUC':>13}")
for split_name, part, metrics in split_metrics:
    print(f"    {split_name:<16}{len(part):>8}{metrics['loss']:>12.4f}"
          f"{metrics['accuracy'] * 100:>12.2f}%{metrics['auc'] * 100:>12.2f}%")
print("-" * W)
print(f"  {'Selisih akurasi latih - uji':<34}: "
      f"{(train_acc_clean - test_acc_clean) * 100:.2f} poin")
print(f"  {'Selisih akurasi validasi - uji':<34}: "
      f"{(val_acc_clean - test_acc_clean) * 100:.2f} poin")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      KINERJA MODEL PADA SUBSET DATA
    ==============================================================================
      Sumber bobot                      : epoch 134
      Ambang klasifikasi                : 0.5
    ------------------------------------------------------------------------------
        Bagian             Citra        Loss     Accuracy      ROC-AUC
        train               2133      0.3065       89.83%       96.87%
        validation           458      0.4297       83.41%       91.13%
        test                 458      0.4116       86.03%       91.80%
    ------------------------------------------------------------------------------
      Selisih akurasi latih - uji       : 3.80 poin
      Selisih akurasi validasi - uji    : -2.62 poin
    ==============================================================================
:::
:::

::: {#6d9cd253 .cell .markdown papermill="{\"duration\":0.119955,\"end_time\":\"2026-08-25T22:31:42.709662+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:42.589707+00:00\",\"status\":\"completed\"}" tags="[]"}
## 6.2 Confusion Matrix {#62-confusion-matrix}
:::

::: {#0f9fd36a .cell .code execution_count="36" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:42.949804Z\",\"iopub.status.busy\":\"2026-08-25T22:31:42.947398Z\",\"iopub.status.idle\":\"2026-08-25T22:31:45.967282Z\",\"shell.execute_reply\":\"2026-08-25T22:31:45.966569Z\"}" papermill="{\"duration\":3.141039,\"end_time\":\"2026-08-25T22:31:45.97019+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:42.829151+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def collect_arrays(dataset):
    """Kumpulkan citra dan label sebenarnya dari sebuah pipeline."""
    images_list, labels_list = [], []
    for batch_images, batch_labels in dataset:
        images_list.append(batch_images.numpy())
        labels_list.append(batch_labels.numpy())
    return np.concatenate(images_list), np.concatenate(labels_list).astype(int)


X_test_eval, y_test_true = collect_arrays(test_ds)

test_probs = final_model.predict(X_test_eval, verbose=0).flatten()
y_test_pred = (test_probs >= THRESHOLD).astype(int)

confusion = confusion_matrix(y_test_true, y_test_pred)
confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
fig.suptitle("Confusion Matrix pada Data Uji", fontsize=12)

panels = [
    (confusion, "d", "Jumlah Citra"),
    (confusion_normalized, ".2f", "Proporsi per Kelas Sebenarnya"),
]

for ax, (matrix, value_format, title) in zip(axes, panels):
    sns.heatmap(matrix, annot=True, fmt=value_format, cmap="Blues", cbar=False,
                square=True, linewidths=0.5, linecolor="white",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title, fontsize=10.5, fontweight="normal")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    ax.grid(False)

plt.tight_layout()
plt.show()

true_negative, false_positive, false_negative, true_positive = confusion.ravel()
n_correct = int(true_negative + true_positive)
n_wrong = int(false_positive + false_negative)

correct_components = [
    ("True Negative  (TN)", f"{CLASS_NAMES[0]} benar", true_negative),
    ("True Positive  (TP)", f"{CLASS_NAMES[1]} benar", true_positive),
]
wrong_components = [
    ("False Positive (FP)", f"{CLASS_NAMES[0]} -> {CLASS_NAMES[1]}",
     false_positive),
    ("False Negative (FN)", f"{CLASS_NAMES[1]} -> {CLASS_NAMES[0]}",
     false_negative),
]

W = 78
print("=" * W)
print("  KOMPONEN CONFUSION MATRIX")
print("=" * W)
print(f"  Kelas positif = {CLASS_NAMES[1]}, "
      f"kelas negatif = {CLASS_NAMES[0]}")
print(f"  Ambang klasifikasi {THRESHOLD} diterapkan pada "
      f"{len(y_test_true)} citra uji")
print("-" * W)
print(f"  Prediksi Benar = {n_correct}")
for code, description, value in correct_components:
    print(f"    {code:<22}{description:<32}: {int(value)}")
print()
print(f"  Prediksi Salah = {n_wrong}")
for code, description, value in wrong_components:
    print(f"    {code:<22}{description:<32}: {int(value)}")
print("=" * W)
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/a97b01f2c8019a298e6eca2c3c0f09908f5dce80.png)
:::

::: {.output .stream .stdout}
    ==============================================================================
      KOMPONEN CONFUSION MATRIX
    ==============================================================================
      Kelas positif = dermatophytosis, kelas negatif = dermatitis
      Ambang klasifikasi 0.5 diterapkan pada 458 citra uji
    ------------------------------------------------------------------------------
      Prediksi Benar = 394
        True Negative  (TN)   dermatitis benar                : 228
        True Positive  (TP)   dermatophytosis benar           : 166

      Prediksi Salah = 64
        False Positive (FP)   dermatitis -> dermatophytosis   : 27
        False Negative (FN)   dermatophytosis -> dermatitis   : 37
    ==============================================================================
:::
:::

::: {#b63af35b .cell .markdown papermill="{\"duration\":0.127113,\"end_time\":\"2026-08-25T22:31:46.222258+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:46.095145+00:00\",\"status\":\"completed\"}" tags="[]"}
## 6.3 Metrik Evaluasi Turunan Confusion Matrix {#63-metrik-evaluasi-turunan-confusion-matrix}
:::

::: {#75100435 .cell .code execution_count="37" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:46.462359Z\",\"iopub.status.busy\":\"2026-08-25T22:31:46.461379Z\",\"iopub.status.idle\":\"2026-08-25T22:31:46.484733Z\",\"shell.execute_reply\":\"2026-08-25T22:31:46.483748Z\"}" papermill="{\"duration\":0.144103,\"end_time\":\"2026-08-25T22:31:46.486197+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:46.342094+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def compute_specificity(matrix):
    """Hitung specificity tiap kelas dari confusion matrix."""
    values = []
    for class_index in range(len(matrix)):
        negatives = (matrix.sum() - matrix[class_index, :].sum()
                     - matrix[:, class_index].sum()
                     + matrix[class_index, class_index])
        false_positives = (matrix[:, class_index].sum()
                           - matrix[class_index, class_index])
        values.append(negatives / (negatives + false_positives))
    return np.array(values)


accuracy = (true_negative + true_positive) / confusion.sum()

precision_per_class = precision_score(y_test_true, y_test_pred, average=None)
recall_per_class = recall_score(y_test_true, y_test_pred, average=None)
f1_per_class = f1_score(y_test_true, y_test_pred, average=None)
specificity_per_class = compute_specificity(confusion)

macro_values = [
    precision_score(y_test_true, y_test_pred, average="macro"),
    recall_score(y_test_true, y_test_pred, average="macro"),
    specificity_per_class.mean(),
    f1_score(y_test_true, y_test_pred, average="macro"),
]

recall_gap = abs(float(recall_per_class[0]) - float(recall_per_class[1]))

W = 78
print("=" * W)
print("  METRIK EVALUASI TURUNAN CONFUSION MATRIX")
print("=" * W)
print(f"  {'Accuracy = (TN + TP) / total':<34}: {accuracy * 100:.2f}%")
print("-" * W)
print("  Metrik per Kelas")
print(f"    {'Kelas':<20}{'Precision':>12}{'Recall':>11}{'Specificity':>14}"
      f"{'F1':>9}")
for class_index, class_name in enumerate(CLASS_NAMES):
    print(f"    {class_name:<20}"
          f"{precision_per_class[class_index] * 100:>11.2f}%"
          f"{recall_per_class[class_index] * 100:>10.2f}%"
          f"{specificity_per_class[class_index] * 100:>13.2f}%"
          f"{f1_per_class[class_index] * 100:>8.2f}%")
print(f"    {'Rata-rata makro':<20}"
      f"{macro_values[0] * 100:>11.2f}%"
      f"{macro_values[1] * 100:>10.2f}%"
      f"{macro_values[2] * 100:>13.2f}%"
      f"{macro_values[3] * 100:>8.2f}%")
print("-" * W)
print(f"  {'Selisih recall antar kelas':<34}: {recall_gap * 100:.2f} poin")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      METRIK EVALUASI TURUNAN CONFUSION MATRIX
    ==============================================================================
      Accuracy = (TN + TP) / total      : 86.03%
    ------------------------------------------------------------------------------
      Metrik per Kelas
        Kelas                  Precision     Recall   Specificity       F1
        dermatitis                86.04%     89.41%        81.77%   87.69%
        dermatophytosis           86.01%     81.77%        89.41%   83.84%
        Rata-rata makro           86.02%     85.59%        85.59%   85.77%
    ------------------------------------------------------------------------------
      Selisih recall antar kelas        : 7.64 poin
    ==============================================================================
:::
:::

::: {#db6e5aa4 .cell .markdown papermill="{\"duration\":0.113412,\"end_time\":\"2026-08-25T22:31:46.712599+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:46.599187+00:00\",\"status\":\"completed\"}" tags="[]"}
## 6.4 Kurva ROC dan AUC {#64-kurva-roc-dan-auc}
:::

::: {#1b3d2c9e .cell .code execution_count="38" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:46.938387Z\",\"iopub.status.busy\":\"2026-08-25T22:31:46.938003Z\",\"iopub.status.idle\":\"2026-08-25T22:31:49.501372Z\",\"shell.execute_reply\":\"2026-08-25T22:31:49.500547Z\"}" papermill="{\"duration\":2.677659,\"end_time\":\"2026-08-25T22:31:49.504126+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:46.826467+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
X_val_eval, y_val_true = collect_arrays(val_ds)
val_probs = final_model.predict(X_val_eval, verbose=0).flatten()

val_fpr, val_tpr, _ = roc_curve(y_val_true, val_probs)

# Seluruh titik dipertahankan agar ambang yang dipakai benar-benar terwakili
test_fpr, test_tpr, test_thresholds = roc_curve(y_test_true, test_probs,
                                                drop_intermediate=False)

val_auc_score = auc(val_fpr, val_tpr)
test_auc_score = auc(test_fpr, test_tpr)

# Ambang dikembalikan dalam urutan menurun, sehingga titik yang mewakili
# aturan "peluang >= THRESHOLD" adalah yang terakhir masih bernilai >= 0.5
threshold_index = int(np.flatnonzero(test_thresholds >= THRESHOLD)[-1])

fig, ax = plt.subplots(figsize=(6.4, 5.6))

ax.plot(test_fpr, test_tpr, color="#4C72B0", lw=1.6,
        label=f"Uji (AUC = {test_auc_score:.4f})")
ax.plot(val_fpr, val_tpr, color="#DD8452", lw=1.4, linestyle="--",
        label=f"Validasi (AUC = {val_auc_score:.4f})")
ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, linestyle=":",
        label="Tebakan acak (AUC = 0.5)")

ax.scatter(test_fpr[threshold_index], test_tpr[threshold_index],
           color="#C62828", s=40, zorder=5,
           label=f"Ambang {THRESHOLD} pada data uji")

ax.set_title("Kurva ROC pada Data Validasi dan Uji", fontsize=11, pad=12)
ax.set_xlabel("False Positive Rate (1 - specificity)")
ax.set_ylabel("True Positive Rate (recall)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

W = 78
print("=" * W)
print("  KURVA ROC DAN AUC")
print("=" * W)
print(f"  {'Kelas positif':<34}: {CLASS_NAMES[1]}")
print(f"  {'Sifat metrik':<34}: tidak bergantung pada ambang")
print("-" * W)
print(f"  {'AUC data validasi':<34}: {val_auc_score * 100:.2f}%")
print(f"  {'AUC data uji':<34}: {test_auc_score * 100:.2f}%")
print(f"  {'Selisih validasi - uji':<34}: "
      f"{(val_auc_score - test_auc_score) * 100:.2f} poin")
print("-" * W)
print(f"  Posisi ambang {THRESHOLD} pada kurva uji")
print(f"  {'  True Positive Rate':<34}: "
      f"{test_tpr[threshold_index] * 100:.2f}%")
print(f"  {'  False Positive Rate':<34}: "
      f"{test_fpr[threshold_index] * 100:.2f}%")
print("=" * W)
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/17a6af43e338736a13c58289440c35383a419288.png)
:::

::: {.output .stream .stdout}
    ==============================================================================
      KURVA ROC DAN AUC
    ==============================================================================
      Kelas positif                     : dermatophytosis
      Sifat metrik                      : tidak bergantung pada ambang
    ------------------------------------------------------------------------------
      AUC data validasi                 : 91.11%
      AUC data uji                      : 91.76%
      Selisih validasi - uji            : -0.66 poin
    ------------------------------------------------------------------------------
      Posisi ambang 0.5 pada kurva uji
        True Positive Rate              : 81.77%
        False Positive Rate             : 10.59%
    ==============================================================================
:::
:::

::: {#40cb1504 .cell .markdown papermill="{\"duration\":0.112807,\"end_time\":\"2026-08-25T22:31:49.749354+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:49.636547+00:00\",\"status\":\"completed\"}" tags="[]"}
# 7. Inference {#7-inference}
:::

::: {#e6dec7d1 .cell .markdown papermill="{\"duration\":0.111153,\"end_time\":\"2026-08-25T22:31:49.973392+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:49.862239+00:00\",\"status\":\"completed\"}" tags="[]"}
## 7.1 Pemrosesan Inferensi {#71-pemrosesan-inferensi}
:::

::: {#cce4f171 .cell .code execution_count="39" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:50.21765Z\",\"iopub.status.busy\":\"2026-08-25T22:31:50.21711Z\",\"iopub.status.idle\":\"2026-08-25T22:31:50.229947Z\",\"shell.execute_reply\":\"2026-08-25T22:31:50.228765Z\"}" papermill="{\"duration\":0.135715,\"end_time\":\"2026-08-25T22:31:50.231559+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:50.095844+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
def preprocess_for_inference(image_path):
    """Siapkan satu berkas citra menjadi masukan siap prediksi.

    Rangkaian langkahnya sama persis dengan yang dikenakan pada data
    latih, sehingga fungsi ini yang harus dipakai ulang saat model
    diterapkan di luar notebook.
    """
    with Image.open(image_path) as img:
        transformed = transform_image(img.convert("RGB"))

    array = np.asarray(transformed, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0), transformed


def predict_single_image(image_path, model=None, threshold=THRESHOLD):
    """Prediksi kelas satu berkas citra beserta tingkat keyakinannya."""
    model = final_model if model is None else model

    batch, preview = preprocess_for_inference(image_path)
    probability = float(model.predict(batch, verbose=0).flatten()[0])

    predicted_index = int(probability >= threshold)
    confidence = probability if predicted_index == 1 else 1.0 - probability

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probability": probability,
        "preview": preview,
    }


W = 78
print("=" * W)
print("  PEMROSESAN INFERENSI")
print("=" * W)
print("  Urutan Langkah")
print(f"  {'  1. Pembacaan berkas':<34}: konversi ke mode RGB")
print(f"  {'  2. Kanonisasi orientasi':<34}: portrait diputar 90 derajat")
print(f"  {'  3. Letterboxing':<34}: kanvas {IMAGE_WIDTH} x {IMAGE_HEIGHT}, "
      f"bantalan hitam")
print(f"  {'  4. Normalisasi':<34}: nilai piksel dibagi 255.0")
print(f"  {'  5. Prediksi':<34}: sigmoid, ambang {THRESHOLD}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      PEMROSESAN INFERENSI
    ==============================================================================
      Urutan Langkah
        1. Pembacaan berkas             : konversi ke mode RGB
        2. Kanonisasi orientasi         : portrait diputar 90 derajat
        3. Letterboxing                 : kanvas 336 x 224, bantalan hitam
        4. Normalisasi                  : nilai piksel dibagi 255.0
        5. Prediksi                     : sigmoid, ambang 0.5
    ==============================================================================
:::
:::

::: {#8ee2c555 .cell .markdown papermill="{\"duration\":0.116632,\"end_time\":\"2026-08-25T22:31:50.461265+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:50.344633+00:00\",\"status\":\"completed\"}" tags="[]"}
## 7.2 Demonstrasi Prediksi pada Data Uji {#72-demonstrasi-prediksi-pada-data-uji}
:::

::: {#301e37a3 .cell .code execution_count="40" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:50.693817Z\",\"iopub.status.busy\":\"2026-08-25T22:31:50.692968Z\",\"iopub.status.idle\":\"2026-08-25T22:31:53.661916Z\",\"shell.execute_reply\":\"2026-08-25T22:31:53.660937Z\"}" papermill="{\"duration\":3.092449,\"end_time\":\"2026-08-25T22:31:53.670837+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:50.578388+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
N_PER_CLASS = 4

# Seed dikembalikan ke nilai global agar pemilihan contoh dapat direproduksi
random.seed(SEED)

for true_label in CLASS_NAMES:
    sample_indices = random.sample(
        test_df[test_df["label"] == true_label].index.tolist(), N_PER_CLASS)

    fig, axes = plt.subplots(1, N_PER_CLASS, figsize=(3.3 * N_PER_CLASS, 3.6))

    for ax, sample_index in zip(axes, sample_indices):
        record = test_df.loc[sample_index]
        result = predict_single_image(record["path"])

        is_correct = result["class"] == true_label
        border_color = "#2E7D32" if is_correct else "#C62828"
        status = "BENAR" if is_correct else "SALAH"

        ax.imshow(result["preview"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"[{status}]\n"
                     f"Prediksi: {result['class']}\n"
                     f"Probabilitas: {result['probability']:.3f}\n"
                     f"Keyakinan: {result['confidence'] * 100:.1f}%",
                     fontsize=8, color=border_color, pad=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

    fig.suptitle(f"Demonstrasi Prediksi pada Data Uji, Kelas {true_label}",
                 fontsize=12)
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/973f84d9feb71f3ed165d6711096cce7f9220451.png)
:::

::: {.output .display_data}
![](vertopal_b1db61492bd94e80a319a62c19f2ae1e/5b70caac7ccf1cbdf7c4a399328cda395828d339.png)
:::
:::

::: {#0d1347cf .cell .markdown papermill="{\"duration\":0.125971,\"end_time\":\"2026-08-25T22:31:53.934598+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:53.808627+00:00\",\"status\":\"completed\"}" tags="[]"}
# 8. Model Export {#8-model-export}
:::

::: {#f9086bd0 .cell .code execution_count="41" execution="{\"iopub.execute_input\":\"2026-08-25T22:31:54.183565Z\",\"iopub.status.busy\":\"2026-08-25T22:31:54.182833Z\",\"iopub.status.idle\":\"2026-08-25T22:31:56.147933Z\",\"shell.execute_reply\":\"2026-08-25T22:31:56.147004Z\"}" papermill="{\"duration\":2.092197,\"end_time\":\"2026-08-25T22:31:56.149761+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:54.057564+00:00\",\"status\":\"completed\"}" tags="[]"}
``` python
VERSION_TAG = "v2-KL-r0-SDS"

MODEL_FILENAME = f"{VERSION_TAG}-dermatitis-dermatophytosis-classification.keras"
CONFIG_FILENAME = f"{VERSION_TAG}-inference-config.json"

model_path = WORKING_DIR / MODEL_FILENAME
config_path = WORKING_DIR / CONFIG_FILENAME

final_model.save(model_path)

preprocessing_steps = [
    "Konversi citra ke mode RGB",
    "Putar 90 derajat bila tinggi lebih besar daripada lebar",
    f"Letterbox ke {IMAGE_WIDTH} x {IMAGE_HEIGHT}, bantalan hitam, bilinear",
    "Bagi nilai piksel dengan 255.0",
]

inference_config = {
    "model_file": MODEL_FILENAME,
    "class_names": CLASS_NAMES,
    "positive_class": CLASS_NAMES[1],
    "threshold": THRESHOLD,
    "canvas_width": IMAGE_WIDTH,
    "canvas_height": IMAGE_HEIGHT,
    "input_shape": list(INPUT_SHAPE),
    "preprocessing": preprocessing_steps,
    "weight_criterion": WEIGHT_CRITERION,
    "weight_epoch": int(best_epoch),
    "training_seed": TRAINING_SEED,
    "test_accuracy": round(float(accuracy), 4),
    "test_roc_auc": round(float(test_auc_score), 4),
    "test_macro_recall": round(float(macro_values[1]), 4),
}

with open(config_path, "w") as config_file:
    json.dump(inference_config, config_file, indent=2)

# Model dimuat ulang dari berkas untuk memastikan hasil ekspor setara
loaded_model = tf.keras.models.load_model(model_path)

n_verify = min(50, len(X_test_eval))
verify_batch = X_test_eval[:n_verify]

pred_memory = final_model.predict(verify_batch, verbose=0).flatten()
pred_loaded = loaded_model.predict(verify_batch, verbose=0).flatten()

diff_export = float(np.max(np.abs(pred_memory - pred_loaded)))
diff_evaluation = float(np.max(np.abs(pred_loaded - test_probs[:n_verify])))
verification_passed = diff_export < 1e-5 and diff_evaluation < 1e-5

# Checkpoint tidak lagi diperlukan setelah model tersimpan dan terverifikasi
checkpoint_removed = False
if verification_passed and CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    checkpoint_removed = True

W = 78
print("=" * W)
print("  EKSPOR MODEL")
print("=" * W)
print(f"  {'Berkas model':<34}: {MODEL_FILENAME}")
print(f"  {'Ukuran berkas':<34}: "
      f"{model_path.stat().st_size / 1024 ** 2:.2f} MB")
print(f"  {'Jumlah parameter':<34}: {final_model.count_params():,}")
print(f"  {'Bobot dari epoch':<34}: {best_epoch} dari {EPOCHS}")
print(f"  {'Berkas konfigurasi':<34}: {CONFIG_FILENAME}")
print("-" * W)
print("  Verifikasi Model Hasil Ekspor")
print(f"  {'  Citra uji yang diverifikasi':<34}: {n_verify}")
print(f"  {'  Status':<34}: "
      f"{'lolos' if verification_passed else 'TIDAK LOLOS'}")
print(f"  {'  Checkpoint dihapus':<34}: "
      f"{'ya' if checkpoint_removed else 'tidak'}")
print("-" * W)
print("  Langkah Prapemrosesan yang Wajib Diikuti Aplikasi Web")
for step_number, step in enumerate(preprocessing_steps, start=1):
    print(f"    {step_number}. {step}")
print("=" * W)
```

::: {.output .stream .stdout}
    ==============================================================================
      EKSPOR MODEL
    ==============================================================================
      Berkas model                      : v2-KL-r0-SDS-dermatitis-dermatophytosis-classification.keras
      Ukuran berkas                     : 8.68 MB
      Jumlah parameter                  : 747,713
      Bobot dari epoch                  : 134 dari 150
      Berkas konfigurasi                : v2-KL-r0-SDS-inference-config.json
    ------------------------------------------------------------------------------
      Verifikasi Model Hasil Ekspor
        Citra uji yang diverifikasi     : 50
        Status                          : lolos
        Checkpoint dihapus              : ya
    ------------------------------------------------------------------------------
      Langkah Prapemrosesan yang Wajib Diikuti Aplikasi Web
        1. Konversi citra ke mode RGB
        2. Putar 90 derajat bila tinggi lebih besar daripada lebar
        3. Letterbox ke 336 x 224, bantalan hitam, bilinear
        4. Bagi nilai piksel dengan 255.0
    ==============================================================================
:::
:::

::: {#cf5d8512 .cell .markdown papermill="{\"duration\":0.138193,\"end_time\":\"2026-08-25T22:31:56.427648+00:00\",\"exception\":false,\"start_time\":\"2026-08-25T22:31:56.289455+00:00\",\"status\":\"completed\"}" tags="[]"}
# 9. Penutup {#9-penutup}
:::

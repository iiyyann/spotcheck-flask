---
jupyter:
  kaggle:
    accelerator: none
    dockerImageVersionId: 28755
    isGpuEnabled: false
    isInternetEnabled: false
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
  nbformat_minor: 4
---

::: {.cell .markdown}
## 1. Introduction {#1-introduction}

### Klasifikasi Penyakit Kulit: Eczema dan Tinea

Eczema dan tinea merupakan dua penyakit kulit yang memiliki manifestasi
klinis serupa, seperti kemerahan, rasa gatal, dan kulit bersisik,
sehingga sulit dibedakan hanya melalui pengamatan visual. Kesalahan
diagnosis antara keduanya dapat menyebabkan pemberian terapi yang tidak
tepat.

Notebook ini membangun model **Convolutional Neural Network (CNN)** yang
dilatih dari awal (*from scratch*) tanpa pendekatan *transfer learning*
untuk mengklasifikasikan citra penyakit kulit eczema dan tinea. Tahapan
pengembangan mencakup eksplorasi data, persiapan data, pembangunan
model, pelatihan, evaluasi, hingga inferensi.

**Dataset:** Eczema and Tinea Skin Disease Dataset (DermNet, via
Kaggle)\
**Task:** Binary Image Classification (Eczema vs Tinea)
:::

::: {.cell .markdown}
## 2. Setup & Configuration {#2-setup--configuration}

Bagian ini mencakup impor seluruh library yang digunakan, konfigurasi
perangkat komputasi (GPU), definisi konstanta global, serta penetapan
seed untuk memastikan hasil yang dapat direproduksi.
:::

::: {.cell .code execution_count="1" execution="{\"iopub.execute_input\":\"2026-07-15T13:13:55.695669Z\",\"iopub.status.busy\":\"2026-07-15T13:13:55.695352Z\",\"iopub.status.idle\":\"2026-07-15T13:14:00.792764Z\",\"shell.execute_reply\":\"2026-07-15T13:14:00.791866Z\",\"shell.execute_reply.started\":\"2026-07-15T13:13:55.695624Z\"}" trusted="true"}
``` python
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Verifikasi struktur folder input Kaggle
INPUT_DIR = Path("/kaggle/input")
for root, dirs, files in os.walk(INPUT_DIR):
    level = root.replace(str(INPUT_DIR), "").count(os.sep)
    indent = "    " * level
    print(f"{indent}{os.path.basename(root)}/")
    if level == 2:
        print(f"{indent}    ({len(files)} files)")

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Label Kelas
CLASS_NAMES = ["eczema", "tinea"]

# Konfigurasi Training
EPOCHS = 100

# Konfigurasi GPU
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")
print(f"TensorFlow version: {tf.__version__}")

# Path Dataset
BASE_DIR   = Path("/kaggle/input/datasets/mreihandirizal/eczema-and-tinea-skin-disease-dataset")
ECZEMA_DIR = BASE_DIR / "eczema"
TINEA_DIR  = BASE_DIR / "tinea"
```

::: {.output .stream .stdout}
    input/
        datasets/
            mreihandirizal/
                (0 files)
                eczema-and-tinea-skin-disease-dataset/
                    tinea/
                    eczema/
    INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
    Number of devices: 2
    TensorFlow version: 2.20.0
:::

::: {.output .stream .stderr}
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1784121240.756998    1582 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13756 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
    I0000 00:00:1784121240.759258    1582 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13756 MB memory:  -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5
:::
:::

::: {.cell .markdown}
## 3. Exploratory Data Analysis (EDA) {#3-exploratory-data-analysis-eda}

Sebelum membangun model, karakteristik dataset perlu dipahami terlebih
dahulu. Bagian ini mencakup pemeriksaan struktur dataset, distribusi
kelas, sampel visual citra, analisis ukuran citra, serta analisis
orientasi citra terhadap kelas. Hasil eksplorasi ini menjadi dasar dalam
menentukan langkah Data Preprocessing pada tahap berikutnya.
:::

::: {.cell .markdown}
### 3.1 Struktur Dataset {#31-struktur-dataset}

Memeriksa jumlah citra per kelas serta format file yang tersedia dalam
dataset.
:::

::: {.cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:00.795280Z\",\"iopub.status.busy\":\"2026-07-15T13:14:00.794769Z\",\"iopub.status.idle\":\"2026-07-15T13:14:00.813572Z\",\"shell.execute_reply\":\"2026-07-15T13:14:00.812799Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:00.795254Z\"}" trusted="true"}
``` python
def get_image_paths(directory):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in Path(directory).iterdir() if p.suffix.lower() in extensions]

eczema_paths = get_image_paths(ECZEMA_DIR)
tinea_paths  = get_image_paths(TINEA_DIR)
all_paths    = eczema_paths + tinea_paths
all_labels   = ["eczema"] * len(eczema_paths) + ["tinea"] * len(tinea_paths)

# Format file
eczema_exts = Counter(p.suffix.lower() for p in eczema_paths)
tinea_exts  = Counter(p.suffix.lower() for p in tinea_paths)

print("=" * 52)
print("  STRUKTUR DATASET")
print("=" * 52)
print(f"  {'Kelas':<12} {'Jumlah':>8}    {'Format File'}")
print("-" * 52)
print(f"  {'Eczema':<12} {len(eczema_paths):>8}    {dict(eczema_exts)}")
print(f"  {'Tinea':<12} {len(tinea_paths):>8}    {dict(tinea_exts)}")
print("-" * 52)
print(f"  {'Total':<12} {len(all_paths):>8}")
print("=" * 52)
```

::: {.output .stream .stdout}
    ====================================================
      STRUKTUR DATASET
    ====================================================
      Kelas          Jumlah    Format File
    ----------------------------------------------------
      Eczema           1122    {'.jpeg': 1122}
      Tinea            1025    {'.jpeg': 1025}
    ----------------------------------------------------
      Total            2147
    ====================================================
:::
:::

::: {.cell .markdown}
### 3.2 Distribusi Kelas {#32-distribusi-kelas}

Visualisasi jumlah citra per kelas untuk mengetahui tingkat keseimbangan
dataset.
:::

::: {.cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:00.814809Z\",\"iopub.status.busy\":\"2026-07-15T13:14:00.814614Z\",\"iopub.status.idle\":\"2026-07-15T13:14:00.943542Z\",\"shell.execute_reply\":\"2026-07-15T13:14:00.942786Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:00.814790Z\"}" trusted="true"}
``` python
counts = [len(eczema_paths), len(tinea_paths)]
colors = ["#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(CLASS_NAMES, counts, color=colors, width=0.5, edgecolor="white")

for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        str(count),
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax.set_title("Distribusi Kelas Dataset", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Kelas", fontsize=11)
ax.set_ylabel("Jumlah Citra", fontsize=11)
ax.set_ylim(0, max(counts) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/91b328a1a7c65c0960d8e5c13cf756bb54d80d82.png)
:::
:::

::: {.cell .markdown}
### 3.3 Sampel Citra per Kelas {#33-sampel-citra-per-kelas}

Menampilkan beberapa sampel citra secara acak dari setiap kelas untuk
mengamati karakteristik visual dataset.
:::

::: {.cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:00.944660Z\",\"iopub.status.busy\":\"2026-07-15T13:14:00.944354Z\",\"iopub.status.idle\":\"2026-07-15T13:14:01.871315Z\",\"shell.execute_reply\":\"2026-07-15T13:14:01.870190Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:00.944619Z\"}" trusted="true"}
``` python
samples = {
    "eczema": random.sample(eczema_paths, 5),
    "tinea" : random.sample(tinea_paths,  5),
}

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sampel Citra per Kelas", fontsize=13, fontweight="bold", y=1.01)

for row_idx, (label, paths) in enumerate(samples.items()):
    for col_idx, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        axes[row_idx][col_idx].imshow(img)
        axes[row_idx][col_idx].set_title(label.capitalize(), fontsize=10, fontweight="bold")
        axes[row_idx][col_idx].axis("off")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/29cd47ddd64c912ca44d61e5ade25e6b12a84875.png)
:::
:::

::: {.cell .markdown}
### 3.4 Analisis Ukuran Citra {#34-analisis-ukuran-citra}

Menganalisis distribusi ukuran citra dalam dataset. Hasil analisis ini
menjadi dasar penentuan ukuran resize yang akan digunakan pada tahap
Data Preparation.
:::

::: {.cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:01.872460Z\",\"iopub.status.busy\":\"2026-07-15T13:14:01.872243Z\",\"iopub.status.idle\":\"2026-07-15T13:14:03.756992Z\",\"shell.execute_reply\":\"2026-07-15T13:14:03.756057Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:01.872439Z\"}" trusted="true"}
``` python
widths, heights = [], []

for path in all_paths:
    try:
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        pass

size_counter = Counter(zip(widths, heights))
most_common_size, most_common_count = size_counter.most_common(1)[0]

print("=" * 52)
print("  ANALISIS UKURAN CITRA")
print("=" * 52)
print(f"  Total Citra Dicek   : {len(widths)}")
print(f"  Min Size            : {min(widths)} x {min(heights)}")
print(f"  Max Size            : {max(widths)} x {max(heights)}")
print(f"  Mean Size           : {int(np.mean(widths))} x {int(np.mean(heights))}")
print(f"  Most Common Size    : {most_common_size[0]} x {most_common_size[1]}  ({most_common_count} citra)")
print("=" * 52)

# Rincian jumlah citra per ukuran (10 ukuran terbanyak)
size_df = pd.DataFrame(
    [(f"{w} x {h}", count) for (w, h), count in size_counter.most_common(10)],
    columns=["Ukuran (px)", "Jumlah Citra"]
)
size_df["Persentase"] = (size_df["Jumlah Citra"] / len(widths) * 100).round(2).astype(str) + "%"

print("\n  10 UKURAN CITRA TERBANYAK")
print("-" * 52)
print(size_df.to_string(index=False))
print("-" * 52)
print(f"  Jumlah ukuran unik dalam dataset : {len(size_counter)}")
print("=" * 52)

# Histogram Width dan Height per kelas
size_label_df = pd.DataFrame({
    "width": widths,
    "height": heights,
    "label": all_labels
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, label in zip(axes, CLASS_NAMES):
    subset = size_label_df[size_label_df["label"] == label]
    ax.hist(subset["width"], bins=30, alpha=0.6, color="#4C72B0", edgecolor="white", label="Width")
    ax.hist(subset["height"], bins=30, alpha=0.6, color="#DD8452", edgecolor="white", label="Height")
    ax.set_title(label.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Ukuran (px)")
    ax.set_ylabel("Jumlah Citra")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Distribusi Width dan Height per Kelas", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    ====================================================
      ANALISIS UKURAN CITRA
    ====================================================
      Total Citra Dicek   : 2147
      Min Size            : 294 x 222
      Max Size            : 720 x 1080
      Mean Size           : 643 x 540
      Most Common Size    : 720 x 472  (603 citra)
    ====================================================

      10 UKURAN CITRA TERBANYAK
    ----------------------------------------------------
    Ukuran (px)  Jumlah Citra Persentase
      720 x 472           603     28.09%
      720 x 480           310     14.44%
      477 x 720           210      9.78%
      480 x 720           166      7.73%
      720 x 477           119      5.54%
      294 x 222            45       2.1%
      720 x 474            37      1.72%
      472 x 720            36      1.68%
      720 x 485            25      1.16%
      720 x 466            22      1.02%
    ----------------------------------------------------
      Jumlah ukuran unik dalam dataset : 132
    ====================================================
:::

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/5c7b25fd07769af55601dfe96d4f2ba536e0bbc5.png)
:::
:::

::: {.cell .markdown}
### 3.5 Analisis Orientasi Citra terhadap Kelas {#35-analisis-orientasi-citra-terhadap-kelas}

Memeriksa apakah terdapat korelasi antara orientasi citra (landscape
atau portrait) maupun ukuran citra dengan kelas penyakit kulit. Analisis
ini penting untuk memastikan model tidak mempelajari pola yang tidak
relevan, seperti orientasi atau ukuran citra, sebagai dasar klasifikasi.
:::

::: {.cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:03.758622Z\",\"iopub.status.busy\":\"2026-07-15T13:14:03.758145Z\",\"iopub.status.idle\":\"2026-07-15T13:14:04.956341Z\",\"shell.execute_reply\":\"2026-07-15T13:14:04.955574Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:03.758595Z\"}" trusted="true"}
``` python
def get_orientation(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w > h:
                return "Landscape"
            elif h > w:
                return "Portrait"
            else:
                return "Square"
    except Exception:
        return "Unknown"

orientation_df = pd.DataFrame({
    "path": all_paths,
    "label": all_labels,
    "orientation": [get_orientation(p) for p in all_paths]
})

cross_count = pd.crosstab(orientation_df["label"], orientation_df["orientation"])
cross_pct   = pd.crosstab(orientation_df["label"], orientation_df["orientation"], normalize="index") * 100

combined = cross_count.astype(str) + " (" + cross_pct.round(1).astype(str) + "%)"
combined["Total"] = cross_count.sum(axis=1)

print("=" * 60)
print("  ORIENTASI CITRA PER KELAS")
print("=" * 60)
print(combined.to_string())
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      ORIENTASI CITRA PER KELAS
    ============================================================
    orientation    Landscape     Portrait  Total
    label                                       
    eczema       763 (68.0%)  359 (32.0%)   1122
    tinea        779 (76.0%)  246 (24.0%)   1025
    ============================================================
:::
:::

::: {.cell .markdown}
### 3.6 Ringkasan Exploratory Data Analysis (EDA) {#36-ringkasan-exploratory-data-analysis-eda}

Ringkasan berikut merangkum seluruh temuan dari tahap EDA sebagai dasar
pengambilan keputusan pada tahap Data Preprocessing, khususnya terkait
strategi resize citra dan potensi bias orientasi terhadap kelas.
:::

::: {.cell .code execution_count="7" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:04.957905Z\",\"iopub.status.busy\":\"2026-07-15T13:14:04.957427Z\",\"iopub.status.idle\":\"2026-07-15T13:14:04.965048Z\",\"shell.execute_reply\":\"2026-07-15T13:14:04.963965Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:04.957860Z\"}" trusted="true"}
``` python
print("=" * 60)
print("  RINGKASAN EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)
print(f"  {'Total Citra':<28}: {len(all_paths)} (Eczema = {len(eczema_paths)}, Tinea = {len(tinea_paths)})")
print(f"  {'Jumlah Ukuran Unik':<28}: {len(size_counter)}")
print(f"  {'Ukuran Paling Umum':<28}: {most_common_size[0]} x {most_common_size[1]} ({most_common_count} citra)")
print(f"  {'Rentang Ukuran':<28}: {min(widths)}x{min(heights)} - {max(widths)}x{max(heights)}")
print("-" * 60)
print("  Temuan Utama:")
print("  - Dataset memiliki 132 ukuran citra unik dengan dua orientasi")
print("    dominan, yaitu landscape dan portrait.")
print("  - Terdapat korelasi orientasi-kelas: kedua kelas didominasi")
print("    citra landscape, namun dengan proporsi berbeda")
print("    (Eczema 68.0% vs Tinea 76.0%).")
print("  - Untuk mempertahankan rasio aspek asli citra dan menghindari")
print("    distorsi objek, strategi resize-with-padding (letterbox)")
print("    dipilih pada tahap Data Preprocessing.")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      RINGKASAN EXPLORATORY DATA ANALYSIS (EDA)
    ============================================================
      Total Citra                 : 2147 (Eczema = 1122, Tinea = 1025)
      Jumlah Ukuran Unik          : 132
      Ukuran Paling Umum          : 720 x 472 (603 citra)
      Rentang Ukuran              : 294x222 - 720x1080
    ------------------------------------------------------------
      Temuan Utama:
      - Dataset memiliki 132 ukuran citra unik dengan dua orientasi
        dominan, yaitu landscape dan portrait.
      - Terdapat korelasi orientasi-kelas: kedua kelas didominasi
        citra landscape, namun dengan proporsi berbeda
        (Eczema 68.0% vs Tinea 76.0%).
      - Untuk mempertahankan rasio aspek asli citra dan menghindari
        distorsi objek, strategi resize-with-padding (letterbox)
        dipilih pada tahap Data Preprocessing.
    ============================================================
:::
:::

::: {.cell .markdown}
## 4. Data Preprocessing {#4-data-preprocessing}

Tahap Data Preprocessing meliputi seluruh proses persiapan data sebelum
digunakan pada tahap Modeling, yaitu pembersihan data (Data Cleaning),
transformasi citra (Data Transformation), pembagian dataset (Data
Splitting), dan augmentasi data (Data Augmentation). Setiap subbagian
dijelaskan secara terpisah agar setiap tahapan pemrosesan data dapat
ditelusuri secara sistematis.
:::

::: {.cell .markdown}
### 4.1 Data Cleaning {#41-data-cleaning}

Sebelum data digunakan pada tahap selanjutnya, dilakukan pemeriksaan
kualitas dataset untuk memastikan seluruh citra dapat digunakan secara
valid dalam proses pelatihan model. Pemeriksaan yang dilakukan mencakup
deteksi duplikat identik, deteksi near-duplicate, serta pemeriksaan
integritas file citra.
:::

::: {.cell .markdown}
#### 4.1.1 Deteksi Duplikat Identik (Exact - MD5) {#411-deteksi-duplikat-identik-exact---md5}

Deteksi duplikat identik dilakukan menggunakan hash MD5 terhadap konten
biner setiap file citra. Citra dengan hash yang identik menunjukkan
salinan yang sama persis sampai ke level byte sehingga berpotensi
menyebabkan kebocoran data (*data leakage*) apabila tersebar pada subset
data yang berbeda setelah pembagian dataset.
:::

::: {.cell .code execution_count="8" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:04.966569Z\",\"iopub.status.busy\":\"2026-07-15T13:14:04.966288Z\",\"iopub.status.idle\":\"2026-07-15T13:14:06.371446Z\",\"shell.execute_reply\":\"2026-07-15T13:14:06.370571Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:04.966535Z\"}" trusted="true"}
``` python
import hashlib

def compute_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

hash_records = [(path, label, compute_hash(path)) for path, label in zip(all_paths, all_labels)]
hash_df = pd.DataFrame(hash_records, columns=["path", "label", "hash"])

duplicate_hashes = hash_df[hash_df.duplicated(subset="hash", keep=False)]
duplicate_groups_exact = duplicate_hashes.groupby("hash")["path"].apply(list)

print("=" * 60)
print("  DETEKSI DUPLIKAT IDENTIK (Exact - MD5)")
print("=" * 60)
print(f"  Total Citra Dicek     : {len(hash_df)}")
print(f"  Total Hash Unik       : {hash_df['hash'].nunique()}")
print(f"  Citra Duplikat        : {len(duplicate_hashes)}")
print(f"  Grup Duplikat         : {len(duplicate_groups_exact)}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      DETEKSI DUPLIKAT IDENTIK (Exact - MD5)
    ============================================================
      Total Citra Dicek     : 2147
      Total Hash Unik       : 2040
      Citra Duplikat        : 207
      Grup Duplikat         : 100
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.1.2 Visualisasi Bukti Duplikat Identik {#412-visualisasi-bukti-duplikat-identik}

Beberapa contoh grup duplikat identik ditampilkan secara visual untuk
memberikan bukti bahwa citra yang terdeteksi benar-benar sama, bukan
hanya kemiripan berdasarkan nama file.
:::

::: {.cell .code execution_count="9" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:06.373017Z\",\"iopub.status.busy\":\"2026-07-15T13:14:06.372615Z\",\"iopub.status.idle\":\"2026-07-15T13:14:07.042295Z\",\"shell.execute_reply\":\"2026-07-15T13:14:07.041234Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:06.372990Z\"}" trusted="true"}
``` python
sample_groups_exact = list(duplicate_groups_exact.items())[:3]

fig, axes = plt.subplots(len(sample_groups_exact), 2, figsize=(8, 4 * len(sample_groups_exact)))

for row_idx, (h, paths) in enumerate(sample_groups_exact):
    for col_idx in range(2):
        img = Image.open(paths[col_idx]).convert("RGB")
        axes[row_idx][col_idx].imshow(img)
        axes[row_idx][col_idx].set_title(paths[col_idx].name, fontsize=9)
        axes[row_idx][col_idx].axis("off")

fig.suptitle("Contoh Pasangan Duplikat Identik", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/53355a7e977f3463888ae88c9fc28f60b01a3576.png)
:::
:::

::: {.cell .markdown}
#### 4.1.3 Penghapusan Duplikat Identik {#413-penghapusan-duplikat-identik}

Dari setiap grup duplikat identik yang ditemukan, disisakan satu citra
representatif dan sisanya dihapus dari dataset untuk mencegah risiko
kebocoran data pada tahap pembagian dataset.
:::

::: {.cell .code execution_count="10" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:07.043618Z\",\"iopub.status.busy\":\"2026-07-15T13:14:07.043332Z\",\"iopub.status.idle\":\"2026-07-15T13:14:07.052462Z\",\"shell.execute_reply\":\"2026-07-15T13:14:07.051555Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:07.043594Z\"}" trusted="true"}
``` python
hash_df_dedup = hash_df.drop_duplicates(subset="hash", keep="first").reset_index(drop=True)
removed_exact_count = len(hash_df) - len(hash_df_dedup)

all_paths  = hash_df_dedup["path"].tolist()
all_labels = hash_df_dedup["label"].tolist()

eczema_count = all_labels.count("eczema")
tinea_count  = all_labels.count("tinea")

print("=" * 60)
print("  HASIL PENGHAPUSAN DUPLIKAT IDENTIK")
print("=" * 60)
print(f"  Citra Sebelum   : {len(hash_df)}")
print(f"  Citra Dihapus   : {removed_exact_count}")
print(f"  Citra Setelah   : {len(all_paths)}")
print("-" * 60)
print(f"  Eczema  : {eczema_count}")
print(f"  Tinea   : {tinea_count}")
print(f"  Total   : {len(all_paths)}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      HASIL PENGHAPUSAN DUPLIKAT IDENTIK
    ============================================================
      Citra Sebelum   : 2147
      Citra Dihapus   : 107
      Citra Setelah   : 2040
    ------------------------------------------------------------
      Eczema  : 1046
      Tinea   : 994
      Total   : 2040
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.1.4 Deteksi Near-Duplicate (Perceptual Hashing) {#414-deteksi-near-duplicate-perceptual-hashing}

Deteksi near-duplicate dilakukan pada dataset yang telah bersih dari
duplikat identik menggunakan metode *perceptual hashing* (phash), yaitu
citra yang secara visual sangat mirip namun tidak identik secara biner,
misalnya akibat resize atau kompresi ulang. Pasangan citra dengan jarak
Hamming di bawah ambang batas tertentu dianggap sebagai kandidat
near-duplicate.
:::

::: {.cell .code execution_count="11" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:07.053780Z\",\"iopub.status.busy\":\"2026-07-15T13:14:07.053453Z\",\"iopub.status.idle\":\"2026-07-15T13:14:24.576101Z\",\"shell.execute_reply\":\"2026-07-15T13:14:24.575246Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:07.053756Z\"}" trusted="true"}
``` python
import imagehash

def compute_phash(path):
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))

phash_records = [(path, label, compute_phash(path)) for path, label in zip(all_paths, all_labels)]
phash_df = pd.DataFrame(phash_records, columns=["path", "label", "phash"])

THRESHOLD = 5
near_dup_pairs = []

phash_list = phash_df["phash"].tolist()
path_list  = phash_df["path"].tolist()

for i in range(len(phash_list)):
    for j in range(i + 1, len(phash_list)):
        distance = phash_list[i] - phash_list[j]
        if distance <= THRESHOLD:
            near_dup_pairs.append((path_list[i], path_list[j], distance))

near_dup_df = pd.DataFrame(near_dup_pairs, columns=["path_a", "path_b", "distance"]).sort_values("distance").reset_index(drop=True)

print("=" * 60)
print("  DETEKSI NEAR-DUPLICATE (Perceptual Hashing)")
print("=" * 60)
print(f"  Threshold Jarak Hamming     : <= {THRESHOLD}")
print(f"  Total Pasangan Terdeteksi   : {len(near_dup_df)}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      DETEKSI NEAR-DUPLICATE (Perceptual Hashing)
    ============================================================
      Threshold Jarak Hamming     : <= 5
      Total Pasangan Terdeteksi   : 20
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.1.5 Visualisasi Bukti Near-Duplicate {#415-visualisasi-bukti-near-duplicate}

Pasangan citra dengan jarak Hamming terkecil ditampilkan secara visual
guna verifikasi kemiripan sebelum diputuskan untuk dihapus dari dataset.
:::

::: {.cell .code execution_count="12" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:24.577463Z\",\"iopub.status.busy\":\"2026-07-15T13:14:24.577095Z\",\"iopub.status.idle\":\"2026-07-15T13:14:26.951924Z\",\"shell.execute_reply\":\"2026-07-15T13:14:26.949535Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:24.577435Z\"}" trusted="true"}
``` python
n_show = min(10, len(near_dup_df))
fig, axes = plt.subplots(n_show, 2, figsize=(8, 4 * n_show))

for row_idx in range(n_show):
    row = near_dup_df.iloc[row_idx]
    img_a = Image.open(row["path_a"]).convert("RGB")
    img_b = Image.open(row["path_b"]).convert("RGB")

    axes[row_idx][0].imshow(img_a)
    axes[row_idx][0].set_title(f"{row['path_a'].name}", fontsize=8)
    axes[row_idx][0].axis("off")

    axes[row_idx][1].imshow(img_b)
    axes[row_idx][1].set_title(f"{row['path_b'].name}\n(distance={row['distance']})", fontsize=8)
    axes[row_idx][1].axis("off")

fig.suptitle("Kandidat Near-Duplicate (Perceptual Hashing)", fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/8c27da78c95dd88246e64eb705b00d9f3e80ceae.png)
:::
:::

::: {.cell .markdown}
#### 4.1.6 Penghapusan Near-Duplicate {#416-penghapusan-near-duplicate}

Berdasarkan hasil verifikasi visual, pasangan near-duplicate yang
terkonfirmasi sebagai citra dari subjek yang sama dikelompokkan
menggunakan algoritma *union-find* agar rantai kemiripan (misalnya citra
A mirip B, dan B mirip C) tergabung dalam satu grup. Dari setiap grup,
disisakan satu citra representatif dan sisanya dihapus dari dataset.
:::

::: {.cell .code execution_count="13" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:26.955511Z\",\"iopub.status.busy\":\"2026-07-15T13:14:26.955135Z\",\"iopub.status.idle\":\"2026-07-15T13:14:26.976818Z\",\"shell.execute_reply\":\"2026-07-15T13:14:26.975654Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:26.955452Z\"}" trusted="true"}
``` python
parent = {p: p for p in all_paths}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    root_x, root_y = find(x), find(y)
    if root_x != root_y:
        parent[root_y] = root_x

for _, row in near_dup_df.iterrows():
    union(row["path_a"], row["path_b"])

groups = {}
for p in all_paths:
    root = find(p)
    groups.setdefault(root, []).append(p)

near_dup_groups = [g for g in groups.values() if len(g) > 1]

paths_to_remove = set()
for g in near_dup_groups:
    for p in g[1:]:
        paths_to_remove.add(p)

current_df = pd.DataFrame({"path": all_paths, "label": all_labels})
current_df = current_df[~current_df["path"].isin(paths_to_remove)].reset_index(drop=True)

all_paths  = current_df["path"].tolist()
all_labels = current_df["label"].tolist()

eczema_count = all_labels.count("eczema")
tinea_count  = all_labels.count("tinea")

print("=" * 60)
print("  HASIL PENGHAPUSAN NEAR-DUPLICATE")
print("=" * 60)
print(f"  Total Grup Near-Duplicate   : {len(near_dup_groups)}")
print(f"  Citra Dihapus               : {len(paths_to_remove)}")
print(f"  Citra Setelah               : {len(all_paths)}")
print("-" * 60)
print(f"  Eczema  : {eczema_count}")
print(f"  Tinea   : {tinea_count}")
print(f"  Total   : {len(all_paths)}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      HASIL PENGHAPUSAN NEAR-DUPLICATE
    ============================================================
      Total Grup Near-Duplicate   : 20
      Citra Dihapus               : 20
      Citra Setelah               : 2020
    ------------------------------------------------------------
      Eczema  : 1037
      Tinea   : 983
      Total   : 2020
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.1.7 Pemeriksaan Integritas File {#417-pemeriksaan-integritas-file}

Pemeriksaan integritas file dilakukan untuk memastikan bahwa setiap
citra dapat dibuka dan dibaca dengan baik tanpa mengalami kerusakan.
Citra yang gagal diverifikasi akan dikeluarkan dari dataset karena
berpotensi menyebabkan kesalahan pada tahap pelatihan model.
:::

::: {.cell .code execution_count="14" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:26.978139Z\",\"iopub.status.busy\":\"2026-07-15T13:14:26.977879Z\",\"iopub.status.idle\":\"2026-07-15T13:14:28.218356Z\",\"shell.execute_reply\":\"2026-07-15T13:14:28.217677Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:26.978116Z\"}" trusted="true"}
``` python
def check_integrity(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

integrity_results = [check_integrity(p) for p in all_paths]
integrity_df = pd.DataFrame({
    "path": all_paths,
    "label": all_labels,
    "is_valid": integrity_results
})

corrupted = integrity_df[~integrity_df["is_valid"]]

# Hapus citra yang gagal diverifikasi, jika ada
if len(corrupted) > 0:
    integrity_df = integrity_df[integrity_df["is_valid"]].reset_index(drop=True)
    all_paths  = integrity_df["path"].tolist()
    all_labels = integrity_df["label"].tolist()

print("=" * 60)
print("  PEMERIKSAAN INTEGRITAS FILE")
print("=" * 60)
print(f"  Total Citra Dicek     : {len(integrity_results)}")
print(f"  Citra Valid           : {sum(integrity_results)}")
print(f"  Citra Rusak/Korup     : {len(corrupted)}")
print(f"  Citra Setelah Cleaning: {len(all_paths)}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      PEMERIKSAAN INTEGRITAS FILE
    ============================================================
      Total Citra Dicek     : 2020
      Citra Valid           : 2020
      Citra Rusak/Korup     : 0
      Citra Setelah Cleaning: 2020
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.1.8 Ringkasan Data Cleaning {#418-ringkasan-data-cleaning}

Ringkasan berikut menyajikan rekapitulasi seluruh tahapan pembersihan
data yang telah dilakukan, mulai dari jumlah citra awal hingga jumlah
citra akhir yang siap digunakan pada tahap Data Preparation.
:::

::: {.cell .code execution_count="15" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:28.219646Z\",\"iopub.status.busy\":\"2026-07-15T13:14:28.219230Z\",\"iopub.status.idle\":\"2026-07-15T13:14:28.226547Z\",\"shell.execute_reply\":\"2026-07-15T13:14:28.225573Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:28.219606Z\"}" trusted="true"}
``` python
final_eczema = all_labels.count("eczema")
final_tinea  = all_labels.count("tinea")
final_total  = len(all_paths)

eczema_pct = final_eczema / final_total * 100
tinea_pct  = final_tinea / final_total * 100

cleaning_steps = [
    ("Citra Awal", 2147),
    ("Duplikat Identik Dihapus", -removed_exact_count),
    ("Near-Duplicate Dihapus", -len(paths_to_remove)),
    ("Citra Rusak/Korup Dihapus", -len(corrupted)),
    ("Citra Akhir", final_total),
]

print("=" * 60)
print("  RINGKASAN DATA CLEANING")
print("=" * 60)
for tahap, jumlah in cleaning_steps:
    print(f"  {tahap:<28}: {jumlah:<10}")
print("-" * 60)
print(f"  {'Distribusi Kelas Akhir':<28}: Eczema = {final_eczema} ({eczema_pct:.1f}%), Tinea = {final_tinea} ({tinea_pct:.1f}%)")
print(f"  {'Total Citra Siap Pakai':<28}: {final_total}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      RINGKASAN DATA CLEANING
    ============================================================
      Citra Awal                  : 2147      
      Duplikat Identik Dihapus    : -107      
      Near-Duplicate Dihapus      : -20       
      Citra Rusak/Korup Dihapus   : 0         
      Citra Akhir                 : 2020      
    ------------------------------------------------------------
      Distribusi Kelas Akhir      : Eczema = 1037 (51.3%), Tinea = 983 (48.7%)
      Total Citra Siap Pakai      : 2020
    ============================================================
:::
:::

::: {.cell .markdown}
### 4.2 Data Transformation {#42-data-transformation}

Transformasi data bertujuan menyesuaikan format, ukuran, dan skala nilai
piksel setiap citra agar seragam dan siap digunakan sebagai masukan
model CNN. Tahapan ini mencakup penyeragaman format citra (RGB),
penentuan ukuran target resize, resize dengan padding (letterbox) untuk
mempertahankan rasio aspek asli, serta normalisasi nilai piksel.
:::

::: {.cell .markdown}
#### 4.2.1 Penyeragaman Format Citra (RGB) {#421-penyeragaman-format-citra-rgb}

Setiap citra diperiksa mode warnanya untuk memastikan seluruh data
berada dalam format RGB tiga kanal sebelum diproses lebih lanjut. Citra
yang belum berformat RGB, misalnya grayscale atau CMYK, akan dikonversi
agar konsisten dengan kebutuhan arsitektur CNN.
:::

::: {.cell .code execution_count="16" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:28.228101Z\",\"iopub.status.busy\":\"2026-07-15T13:14:28.227732Z\",\"iopub.status.idle\":\"2026-07-15T13:14:29.440523Z\",\"shell.execute_reply\":\"2026-07-15T13:14:29.439521Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:28.228062Z\"}" trusted="true"}
``` python
def check_image_mode(path):
    with Image.open(path) as img:
        return img.mode

mode_counts = Counter(check_image_mode(p) for p in all_paths)
non_rgb_count = sum(count for mode, count in mode_counts.items() if mode != "RGB")

print("=" * 60)
print("  PEMERIKSAAN FORMAT CITRA (MODE WARNA)")
print("=" * 60)
for mode, count in mode_counts.items():
    print(f"  {mode:<28}: {count}")
print("-" * 60)
print(f"  {'Total Citra':<28}: {len(all_paths)}")
print(f"  {'Citra Non-RGB':<28}: {non_rgb_count}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      PEMERIKSAAN FORMAT CITRA (MODE WARNA)
    ============================================================
      RGB                         : 2020
    ------------------------------------------------------------
      Total Citra                 : 2020
      Citra Non-RGB               : 0
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.2.2 Penentuan Ukuran Target Resize {#422-penentuan-ukuran-target-resize}

Berdasarkan hasil analisis ukuran citra pada tahap EDA, ukuran target
resize ditentukan sebesar 224x224 piksel. Ukuran ini dipilih karena
merupakan standar umum pada arsitektur CNN serta konsisten dengan ukuran
yang digunakan pada eksperimen model sebelumnya (Model A hingga Model
F), sehingga performa model baru tetap dapat dibandingkan secara setara.
:::

::: {.cell .code execution_count="17" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:29.442407Z\",\"iopub.status.busy\":\"2026-07-15T13:14:29.441560Z\",\"iopub.status.idle\":\"2026-07-15T13:14:29.448210Z\",\"shell.execute_reply\":\"2026-07-15T13:14:29.447329Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:29.442381Z\"}" trusted="true"}
``` python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("=" * 60)
print("  KONFIGURASI UKURAN CITRA")
print("=" * 60)
print(f"  {'Ukuran Target (Resize)':<28}: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
print(f"  {'Batch Size':<28}: {BATCH_SIZE}")
print("-" * 60)
print("  Alasan Pemilihan:")
print("  - Konsisten dengan ukuran pada eksperimen Model A - F.")
print("  - Ukuran standar untuk custom CNN training-from-scratch.")
print("  - Menyeimbangkan detail visual dengan efisiensi komputasi.")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      KONFIGURASI UKURAN CITRA
    ============================================================
      Ukuran Target (Resize)      : 224 x 224
      Batch Size                  : 32
    ------------------------------------------------------------
      Alasan Pemilihan:
      - Konsisten dengan ukuran pada eksperimen Model A - F.
      - Ukuran standar untuk custom CNN training-from-scratch.
      - Menyeimbangkan detail visual dengan efisiensi komputasi.
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.2.3 Resize dengan Padding (Letterbox) {#423-resize-dengan-padding-letterbox}

Berbeda dengan resize biasa yang meregangkan citra secara langsung ke
ukuran target sehingga dapat mengubah rasio aspek objek, teknik
letterbox mengubah ukuran citra secara proporsional kemudian menambahkan
padding pada sisi yang lebih pendek agar hasil akhir berbentuk persegi.
Pendekatan ini dipilih untuk mempertahankan bentuk asli objek pada
citra, mengingat temuan EDA menunjukkan adanya dua orientasi dominan
(landscape dan portrait) dalam dataset.
:::

::: {.cell .code execution_count="18" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:29.449362Z\",\"iopub.status.busy\":\"2026-07-15T13:14:29.449145Z\",\"iopub.status.idle\":\"2026-07-15T13:14:29.469821Z\",\"shell.execute_reply\":\"2026-07-15T13:14:29.469159Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:29.449341Z\"}" trusted="true"}
``` python
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


# Demonstrasi pada satu sampel citra
sample_path = all_paths[0]
with Image.open(sample_path) as sample_img:
    sample_img = sample_img.convert("RGB")
    original_size = sample_img.size
    letterboxed_img = letterbox_resize(sample_img, IMAGE_SIZE)

print("=" * 60)
print("  DEMONSTRASI LETTERBOX RESIZE")
print("=" * 60)
print(f"  {'Ukuran Asli':<28}: {original_size[0]} x {original_size[1]}")
print(f"  {'Ukuran Setelah Letterbox':<28}: {letterboxed_img.size[0]} x {letterboxed_img.size[1]}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      DEMONSTRASI LETTERBOX RESIZE
    ============================================================
      Ukuran Asli                 : 720 x 472
      Ukuran Setelah Letterbox    : 224 x 224
    ============================================================
:::
:::

::: {.cell .markdown}
#### 4.2.4 Visualisasi Hasil Resize {#424-visualisasi-hasil-resize}

Perbandingan visual antara citra asli, hasil resize biasa (stretch), dan
hasil letterbox resize ditampilkan untuk membuktikan bahwa letterbox
resize berhasil mempertahankan rasio aspek objek tanpa distorsi,
khususnya pada citra dengan orientasi portrait yang paling rentan
mengalami perubahan bentuk akibat resize biasa.
:::

::: {.cell .code execution_count="19" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:29.471004Z\",\"iopub.status.busy\":\"2026-07-15T13:14:29.470714Z\",\"iopub.status.idle\":\"2026-07-15T13:14:30.257149Z\",\"shell.execute_reply\":\"2026-07-15T13:14:30.253910Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:29.470972Z\"}" trusted="true"}
``` python
# Ambil sampel citra portrait dan landscape untuk perbandingan
sample_orientations = {
    "Portrait": next(p for p in all_paths if Image.open(p).size[1] > Image.open(p).size[0]),
    "Landscape": next(p for p in all_paths if Image.open(p).size[0] > Image.open(p).size[1]),
}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Perbandingan Resize Biasa vs Letterbox Resize", fontsize=13, fontweight="bold")

for row_idx, (orientation, path) in enumerate(sample_orientations.items()):
    with Image.open(path) as img:
        img = img.convert("RGB")
        stretched_img  = img.resize(IMAGE_SIZE, Image.BILINEAR)
        letterboxed_img = letterbox_resize(img, IMAGE_SIZE)

        for col_idx, (title, display_img) in enumerate([
            (f"Asli ({orientation})", img),
            ("Resize Biasa (Stretch)", stretched_img),
            ("Letterbox Resize", letterboxed_img),
        ]):
            axes[row_idx][col_idx].imshow(display_img)
            axes[row_idx][col_idx].set_title(title, fontsize=10)
            axes[row_idx][col_idx].axis("off")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/4f78ae108232af04129428c99eade0362abb9d4d.png)
:::
:::

::: {.cell .markdown}
#### 4.2.5 Normalisasi Piksel {#425-normalisasi-piksel}

Setelah proses resize, nilai piksel citra yang semula berada pada
rentang 0-255 dinormalisasi ke rentang 0-1 dengan cara membaginya dengan

1.  Normalisasi ini bertujuan menjaga kestabilan proses pelatihan serta
    mempercepat konvergensi algoritma optimasi.
:::

::: {.cell .code execution_count="20" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:30.258419Z\",\"iopub.status.busy\":\"2026-07-15T13:14:30.258199Z\",\"iopub.status.idle\":\"2026-07-15T13:14:30.266058Z\",\"shell.execute_reply\":\"2026-07-15T13:14:30.265294Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:30.258399Z\"}" trusted="true"}
``` python
sample_array = img_to_array(letterboxed_img)
normalized_array = sample_array / 255.0

print("=" * 60)
print("  DEMONSTRASI NORMALISASI PIKSEL")
print("=" * 60)
print(f"  {'Rentang Sebelum Normalisasi':<28}: {sample_array.min():.1f} - {sample_array.max():.1f}")
print(f"  {'Rentang Setelah Normalisasi':<28}: {normalized_array.min():.2f} - {normalized_array.max():.2f}")
print(f"  {'Shape Array':<28}: {normalized_array.shape}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      DEMONSTRASI NORMALISASI PIKSEL
    ============================================================
      Rentang Sebelum Normalisasi : 0.0 - 228.0
      Rentang Setelah Normalisasi : 0.00 - 0.89
      Shape Array                 : (224, 224, 3)
    ============================================================
:::
:::

::: {.cell .markdown}
### 4.3 Data Splitting {#43-data-splitting}

Dataset yang telah melalui tahap Data Cleaning dibagi menjadi data latih
(train), data validasi (validation), dan data uji (test) dengan rasio
70%, 15%, dan 15% menggunakan stratified split. Stratifikasi memastikan
proporsi kelas eczema dan tinea tetap konsisten pada setiap subset,
sehingga proses pelatihan dan evaluasi model dapat dilakukan secara
representatif.
:::

::: {.cell .code execution_count="21" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:30.267631Z\",\"iopub.status.busy\":\"2026-07-15T13:14:30.266935Z\",\"iopub.status.idle\":\"2026-07-15T13:14:30.293996Z\",\"shell.execute_reply\":\"2026-07-15T13:14:30.293055Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:30.267593Z\"}" trusted="true"}
``` python
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15

df = pd.DataFrame({"path": all_paths, "label": all_labels})

# Split train (70%) dan temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_SPLIT + TEST_SPLIT),
    stratify=df["label"],
    random_state=SEED
)

# Split temp menjadi validation (15%) dan test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)


def split_summary(split_df, name):
    counts = split_df["label"].value_counts()
    total  = len(split_df)
    eczema_n = counts.get("eczema", 0)
    tinea_n  = counts.get("tinea", 0)
    return {
        "Split"  : name,
        "Total"  : total,
        "Eczema" : f"{eczema_n} ({eczema_n / total * 100:.1f}%)",
        "Tinea"  : f"{tinea_n} ({tinea_n / total * 100:.1f}%)",
    }

summary = pd.DataFrame([
    split_summary(train_df, "Train"),
    split_summary(val_df,   "Validation"),
    split_summary(test_df,  "Test"),
])

print("=" * 60)
print("  SPLIT DATASET")
print("=" * 60)
print(f"  {'Split':<12}{'Total':<8}{'Eczema':<16}{'Tinea':<16}")
print("-" * 60)
for _, row in summary.iterrows():
    print(f"  {row['Split']:<12}{row['Total']:<8}{row['Eczema']:<16}{row['Tinea']:<16}")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      SPLIT DATASET
    ============================================================
      Split       Total   Eczema          Tinea           
    ------------------------------------------------------------
      Train       1414    726 (51.3%)     688 (48.7%)     
      Validation  303     156 (51.5%)     147 (48.5%)     
      Test        303     155 (51.2%)     148 (48.8%)     
    ============================================================
:::
:::

::: {.cell .markdown}
### 4.4 Data Augmentation {#44-data-augmentation}

Augmentasi data diterapkan untuk memperluas variasi citra latih tanpa
mengubah label kelasnya, sehingga model mampu mempelajari karakteristik
objek dari berbagai variasi kondisi dan risiko overfitting dapat
dikurangi. Berdasarkan temuan pada eksperimen sebelumnya, augmentasi
tingkat sedang (horizontal dan vertical flip, rotasi, penyesuaian
brightness, serta zoom sederhana) terbukti memberikan generalisasi
terbaik dibandingkan augmentasi agresif, sehingga strategi ini dipilih
sebagai pendekatan utama. Augmentasi hanya diterapkan pada data latih,
sedangkan data validasi dan data uji hanya melalui resize dan
normalisasi tanpa augmentasi agar evaluasi mencerminkan kondisi data
asli.
:::

::: {.cell .markdown}
#### 4.4.1 Custom Data Generator {#441-custom-data-generator}

Data generator kustom dibangun menggunakan `tf.keras.utils.Sequence`
untuk memuat citra secara batch, menerapkan resize dengan padding
(letterbox), normalisasi piksel, serta augmentasi pada data latih.
Pendekatan kustom digunakan agar proses letterbox resize dapat
diterapkan secara konsisten pada seluruh subset data, menggantikan
fungsi bawaan Keras yang secara default melakukan resize dengan
peregangan (stretch).
:::

::: {.cell .code execution_count="22" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:30.295424Z\",\"iopub.status.busy\":\"2026-07-15T13:14:30.295098Z\",\"iopub.status.idle\":\"2026-07-15T13:14:30.307149Z\",\"shell.execute_reply\":\"2026-07-15T13:14:30.306432Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:30.295386Z\"}" trusted="true"}
``` python
class SkinDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, image_size, batch_size, augment=False, shuffle=True):
        super().__init__()
        self.df         = df.copy()
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment    = augment
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for _, row in batch.iterrows():
            with Image.open(row["path"]) as img:
                img = img.convert("RGB")
                img = letterbox_resize(img, self.image_size)
                img = img_to_array(img) / 255.0

            if self.augment:
                img = self._augment(img)

            images.append(img)
            labels.append(0 if row["label"] == "eczema" else 1)

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    def _augment(self, img):
        # Horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        # Vertical flip
        if random.random() > 0.5:
            img = np.flipud(img)
        # Rotasi acak (0, 90, 180, 270 derajat)
        k = random.randint(0, 3)
        img = np.rot90(img, k)
        # Brightness adjustment ringan
        factor = random.uniform(0.85, 1.15)
        img = np.clip(img * factor, 0, 1)
        # Zoom sederhana via crop dan resize
        if random.random() > 0.5:
            h, w = img.shape[:2]
            crop = random.uniform(0.9, 1.0)
            ch, cw = int(h * crop), int(w * crop)
            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
            img = img[y:y+ch, x:x+cw]
            img = np.array(
                Image.fromarray((img * 255).astype(np.uint8)).resize((w, h))
            ) / 255.0
        return img
```
:::

::: {.cell .markdown}
#### 4.4.2 Visualisasi Hasil Augmentasi {#442-visualisasi-hasil-augmentasi}

Beberapa hasil augmentasi ditampilkan untuk masing-masing kelas guna
membuktikan bahwa variasi yang dihasilkan tetap mempertahankan
karakteristik visual utama objek, tanpa distorsi berlebihan yang dapat
menghilangkan informasi penting untuk klasifikasi.
:::

::: {.cell .code execution_count="23" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:30.308402Z\",\"iopub.status.busy\":\"2026-07-15T13:14:30.308120Z\",\"iopub.status.idle\":\"2026-07-15T13:14:31.318458Z\",\"shell.execute_reply\":\"2026-07-15T13:14:31.317330Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:30.308379Z\"}" trusted="true"}
``` python
eczema_sample_df = train_df[train_df["label"] == "eczema"].iloc[:1]
tinea_sample_df  = train_df[train_df["label"] == "tinea"].iloc[:1]

eczema_gen = SkinDataGenerator(eczema_sample_df, IMAGE_SIZE, batch_size=1, augment=True, shuffle=False)
tinea_gen  = SkinDataGenerator(tinea_sample_df,  IMAGE_SIZE, batch_size=1, augment=True, shuffle=False)

fig, axes = plt.subplots(2, 6, figsize=(16, 6))
fig.suptitle("Contoh Hasil Augmentasi per Kelas (Citra Sama, 6 Variasi)", fontsize=13, fontweight="bold")

for col_idx in range(6):
    eczema_img, _ = eczema_gen[0]
    axes[0][col_idx].imshow(eczema_img[0])
    axes[0][col_idx].set_title(f"Eczema - Variasi {col_idx + 1}", fontsize=9)
    axes[0][col_idx].axis("off")

    tinea_img, _ = tinea_gen[0]
    axes[1][col_idx].imshow(tinea_img[0])
    axes[1][col_idx].set_title(f"Tinea - Variasi {col_idx + 1}", fontsize=9)
    axes[1][col_idx].axis("off")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/35499d60ea990d2c4cc29f4a7dd4f1dc5774dbeb.png)
:::
:::

::: {.cell .markdown}
#### 4.4.3 Inisialisasi Data Generator {#443-inisialisasi-data-generator}

Data generator diinisialisasi untuk masing-masing subset. Generator data
latih menerapkan augmentasi dan pengacakan urutan data, sedangkan
generator data validasi dan data uji hanya melakukan resize dan
normalisasi tanpa augmentasi maupun pengacakan, agar evaluasi model
konsisten dan dapat direproduksi.
:::

::: {.cell .code execution_count="24" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:31.320882Z\",\"iopub.status.busy\":\"2026-07-15T13:14:31.320057Z\",\"iopub.status.idle\":\"2026-07-15T13:14:31.329656Z\",\"shell.execute_reply\":\"2026-07-15T13:14:31.328901Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:31.320842Z\"}" trusted="true"}
``` python
train_gen = SkinDataGenerator(train_df, IMAGE_SIZE, BATCH_SIZE, augment=True,  shuffle=True)
val_gen   = SkinDataGenerator(val_df,   IMAGE_SIZE, BATCH_SIZE, augment=False, shuffle=False)
test_gen  = SkinDataGenerator(test_df,  IMAGE_SIZE, BATCH_SIZE, augment=False, shuffle=False)

print("=" * 60)
print("  DATA GENERATOR")
print("=" * 60)
print(f"  {'Train Batches':<28}: {len(train_gen)}")
print(f"  {'Validation Batches':<28}: {len(val_gen)}")
print(f"  {'Test Batches':<28}: {len(test_gen)}")
print("-" * 60)
print(f"  {'Ukuran Citra':<28}: {IMAGE_SIZE}")
print(f"  {'Batch Size':<28}: {BATCH_SIZE}")
print(f"  {'Augmentasi Train':<28}: Flip, Rotasi, Brightness, Zoom")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      DATA GENERATOR
    ============================================================
      Train Batches               : 45
      Validation Batches          : 10
      Test Batches                : 10
    ------------------------------------------------------------
      Ukuran Citra                : (224, 224)
      Batch Size                  : 32
      Augmentasi Train            : Flip, Rotasi, Brightness, Zoom
    ============================================================
:::
:::

::: {.cell .markdown}
## 5. Modeling {#5-modeling}

Bagian ini membahas tahap pembangunan model klasifikasi menggunakan
arsitektur Convolutional Neural Network (CNN) yang dibangun dan dilatih
secara training from scratch. Model dikembangkan berdasarkan arsitektur
terbaik dari hasil eksperimen sebelumnya, dengan penyesuaian strategi
learning rate schedule dan fungsi loss untuk mendorong performa yang
lebih baik.
:::

::: {.cell .markdown}
### 5.1 Strategi Pengembangan Model {#51-strategi-pengembangan-model}

Berdasarkan hasil perbandingan pada eksperimen sebelumnya, arsitektur
Model F (Residual CNN dengan skip connection dan learning rate warmup)
memberikan performa terbaik dengan F1-Score sebesar 81.01% dan AUC-ROC
sebesar 87.91% pada test set. Oleh karena itu, arsitektur Model F
dijadikan dasar pengembangan model final pada penelitian ini.

Dua penyesuaian diterapkan untuk mendorong performa lebih tinggi:

1.  Cosine Annealing Learning Rate Schedule menggantikan kombinasi
    learning rate warmup manual dan ReduceLROnPlateau. Learning rate
    dimulai sangat kecil, naik bertahap menuju nilai target selama fase
    warmup, kemudian menurun secara halus mengikuti kurva kosinus hingga
    mendekati nol pada akhir pelatihan. Pendekatan ini menghasilkan
    penurunan learning rate yang lebih stabil dibandingkan penurunan
    bertahap (step-wise) pada ReduceLROnPlateau.
2.  Label Smoothing diterapkan pada fungsi loss dengan nilai 0.1 untuk
    mengurangi tingkat kepercayaan model yang berlebihan terhadap label
    target, sehingga membantu meningkatkan kemampuan generalisasi model.

Arsitektur inti (lima blok konvolusi dengan residual connection serta
SeparableConv2D pada blok terakhir) dipertahankan tanpa perubahan
struktural, mengingat arsitektur tersebut telah terbukti memberikan
performa terbaik pada eksperimen sebelumnya.
:::

::: {.cell .markdown}
### 5.2 Arsitektur Model {#52-arsitektur-model}

Arsitektur dibangun menggunakan Functional API Keras untuk mendukung
residual connection. Model terdiri atas lima blok konvolusi dengan skip
connection pada blok ketiga-keempat dan blok kelima-keenam, Global
Average Pooling, serta dua lapisan Dense (256 -\> 128) sebelum lapisan
output.
:::

::: {.cell .code execution_count="25" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:31.331399Z\",\"iopub.status.busy\":\"2026-07-15T13:14:31.330841Z\",\"iopub.status.idle\":\"2026-07-15T13:14:31.346141Z\",\"shell.execute_reply\":\"2026-07-15T13:14:31.345417Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:31.331375Z\"}" trusted="true"}
``` python
def build_model(input_shape, num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    # Blok 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Blok 3 + Blok 4 dengan residual connection
    residual = layers.Conv2D(128, (1, 1), padding="same")(x)
    residual = layers.MaxPooling2D((2, 2))(residual)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)

    # Blok 5 + Blok 6 dengan residual connection
    residual2 = layers.Conv2D(256, (1, 1), padding="same")(x)
    residual2 = layers.MaxPooling2D((2, 2))(residual2)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, residual2])
    x = layers.Activation("relu")(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="Final_Residual_CNN")
    return model
```
:::

::: {.cell .markdown}
### 5.3 Kompilasi Model {#53-kompilasi-model}

Model dikompilasi menggunakan optimizer Adam dengan learning rate
schedule Cosine Annealing yang mencakup fase warmup. Learning rate
dimulai dari 1e-5, naik secara bertahap menuju target 5e-4 selama 10
epoch pertama, kemudian menurun mengikuti kurva kosinus hingga mendekati
nol pada epoch ke-100. Fungsi loss Binary Crossentropy diterapkan dengan
label smoothing sebesar 0.1 untuk mengurangi overconfidence model
terhadap label target.
:::

::: {.cell .code execution_count="26" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:31.347367Z\",\"iopub.status.busy\":\"2026-07-15T13:14:31.347150Z\",\"iopub.status.idle\":\"2026-07-15T13:14:32.452191Z\",\"shell.execute_reply\":\"2026-07-15T13:14:32.451550Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:31.347348Z\"}" trusted="true"}
``` python
WARMUP_EPOCHS   = 10
TARGET_LR       = 5e-4
INITIAL_LR      = 1e-5
LABEL_SMOOTHING = 0.1

steps_per_epoch = len(train_gen)
warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
total_steps     = EPOCHS * steps_per_epoch
decay_steps     = total_steps - warmup_steps

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=decay_steps,
    alpha=0.0,
    warmup_target=TARGET_LR,
    warmup_steps=warmup_steps
)

with strategy.scope():
    model = build_model(input_shape=(*IMAGE_SIZE, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )

model.summary()
```

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "Final_Residual_CNN"</span>
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">896</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalization │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">128</span> │ conv2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>,  │     <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │ max_pooling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_1     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │ max_pooling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_3     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ max_pooling2d_3[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">8,320</span> │ max_pooling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │ max_pooling2d_2[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │    <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │ activation[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_5     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ separable_conv2d    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">68,096</span> │ max_pooling2d_5[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SeparableConv2D</span>)   │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">33,024</span> │ activation[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ separable_conv2d… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_4     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │ max_pooling2d_4[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_1        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
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
:::

::: {.cell .markdown}
### 5.4 Callbacks {#54-callbacks}

EarlyStopping digunakan untuk menghentikan pelatihan apabila val_loss
tidak menunjukkan perbaikan setelah 15 epoch berturut-turut, sekaligus
mengembalikan bobot terbaik yang diperoleh selama pelatihan.
ModelCheckpoint digunakan untuk menyimpan bobot model dengan val_loss
terendah. Penurunan learning rate tidak lagi memerlukan
ReduceLROnPlateau karena telah ditangani melalui Cosine Annealing
Learning Rate Schedule pada tahap kompilasi model.
:::

::: {.cell .code execution_count="27" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:32.453602Z\",\"iopub.status.busy\":\"2026-07-15T13:14:32.453237Z\",\"iopub.status.idle\":\"2026-07-15T13:14:32.458174Z\",\"shell.execute_reply\":\"2026-07-15T13:14:32.457234Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:32.453566Z\"}" trusted="true"}
``` python
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=0
    ),
    ModelCheckpoint(
        filepath="model_final_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )
]
```
:::

::: {.cell .markdown}
### 5.5 Pelatihan Model {#55-pelatihan-model}

Model dilatih menggunakan data generator dengan augmentasi tingkat
sedang pada data latih. Proses pelatihan berjalan hingga maksimal 100
epoch atau berhenti lebih awal apabila val_loss tidak membaik sesuai
konfigurasi EarlyStopping.
:::

::: {.cell .code execution_count="28" execution="{\"iopub.execute_input\":\"2026-07-15T13:14:32.459651Z\",\"iopub.status.busy\":\"2026-07-15T13:14:32.459177Z\",\"iopub.status.idle\":\"2026-07-15T13:37:54.770813Z\",\"shell.execute_reply\":\"2026-07-15T13:37:54.769987Z\",\"shell.execute_reply.started\":\"2026-07-15T13:14:32.459590Z\"}" trusted="true"}
``` python
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

stopped_epoch = len(history.history["loss"])
best_epoch    = int(np.argmin(history.history["val_loss"])) + 1
train_acc     = history.history["accuracy"][best_epoch - 1]
train_loss    = history.history["loss"][best_epoch - 1]
val_acc       = history.history["val_accuracy"][best_epoch - 1]
val_loss      = history.history["val_loss"][best_epoch - 1]

print("=" * 60)
print("  TRAINING SUMMARY")
print("=" * 60)
print(f"  {'Stopped at Epoch':<28}: {stopped_epoch} / {EPOCHS}")
print(f"  {'Best Epoch':<28}: {best_epoch}")
print(f"  {'Train Accuracy':<28}: {train_acc * 100:.2f}%")
print(f"  {'Train Loss':<28}: {train_loss:.4f}")
print(f"  {'Val Accuracy':<28}: {val_acc * 100:.2f}%")
print(f"  {'Val Loss':<28}: {val_loss:.4f}")
print("=" * 60)
```

::: {.output .stream .stdout}
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    Epoch 1/100
    INFO:tensorflow:Collective all_reduce tensors: 35 all_reduces, num_devices = 2, group_size = 2, implementation = CommunicationImplementation.NCCL, num_packs = 1
    45/45 ━━━━━━━━━━━━━━━━━━━━ 0s 273ms/step - accuracy: 0.4966 - loss: 0.7409INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    45/45 ━━━━━━━━━━━━━━━━━━━━ 28s 379ms/step - accuracy: 0.5219 - loss: 0.7179 - val_accuracy: 0.5149 - val_loss: 0.6928
    Epoch 2/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 373ms/step - accuracy: 0.6280 - loss: 0.6651 - val_accuracy: 0.5149 - val_loss: 0.6927
    Epoch 3/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 366ms/step - accuracy: 0.6443 - loss: 0.6447 - val_accuracy: 0.5149 - val_loss: 0.6930
    Epoch 4/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 379ms/step - accuracy: 0.6259 - loss: 0.6387 - val_accuracy: 0.5149 - val_loss: 0.6931
    Epoch 5/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 371ms/step - accuracy: 0.6612 - loss: 0.6291 - val_accuracy: 0.5149 - val_loss: 0.6917
    Epoch 6/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 369ms/step - accuracy: 0.6846 - loss: 0.6160 - val_accuracy: 0.4884 - val_loss: 0.7043
    Epoch 7/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 369ms/step - accuracy: 0.6584 - loss: 0.6204 - val_accuracy: 0.5248 - val_loss: 0.6943
    Epoch 8/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 359ms/step - accuracy: 0.6648 - loss: 0.6148 - val_accuracy: 0.5182 - val_loss: 0.7195
    Epoch 9/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 366ms/step - accuracy: 0.7001 - loss: 0.6008 - val_accuracy: 0.5248 - val_loss: 0.7318
    Epoch 10/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 363ms/step - accuracy: 0.6839 - loss: 0.6124 - val_accuracy: 0.5281 - val_loss: 0.7012
    Epoch 11/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 359ms/step - accuracy: 0.6924 - loss: 0.6066 - val_accuracy: 0.5149 - val_loss: 0.7622
    Epoch 12/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 358ms/step - accuracy: 0.6917 - loss: 0.5943 - val_accuracy: 0.5380 - val_loss: 0.7114
    Epoch 13/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 363ms/step - accuracy: 0.6825 - loss: 0.6051 - val_accuracy: 0.6733 - val_loss: 0.6186
    Epoch 14/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 369ms/step - accuracy: 0.7065 - loss: 0.5791 - val_accuracy: 0.6535 - val_loss: 0.6408
    Epoch 15/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 366ms/step - accuracy: 0.6909 - loss: 0.6008 - val_accuracy: 0.6964 - val_loss: 0.6075
    Epoch 16/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.7143 - loss: 0.5825 - val_accuracy: 0.6172 - val_loss: 0.6329
    Epoch 17/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 362ms/step - accuracy: 0.7072 - loss: 0.5824 - val_accuracy: 0.6931 - val_loss: 0.5941
    Epoch 18/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.7115 - loss: 0.5915 - val_accuracy: 0.6370 - val_loss: 0.6433
    Epoch 19/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 363ms/step - accuracy: 0.7284 - loss: 0.5818 - val_accuracy: 0.5710 - val_loss: 0.8262
    Epoch 20/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 365ms/step - accuracy: 0.7298 - loss: 0.5675 - val_accuracy: 0.7492 - val_loss: 0.5602
    Epoch 21/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 364ms/step - accuracy: 0.7270 - loss: 0.5735 - val_accuracy: 0.7789 - val_loss: 0.5396
    Epoch 22/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 367ms/step - accuracy: 0.7122 - loss: 0.5802 - val_accuracy: 0.7855 - val_loss: 0.5324
    Epoch 23/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 369ms/step - accuracy: 0.7348 - loss: 0.5588 - val_accuracy: 0.7525 - val_loss: 0.5311
    Epoch 24/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 361ms/step - accuracy: 0.7397 - loss: 0.5504 - val_accuracy: 0.6832 - val_loss: 0.6210
    Epoch 25/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 363ms/step - accuracy: 0.7362 - loss: 0.5602 - val_accuracy: 0.6898 - val_loss: 0.5986
    Epoch 26/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 370ms/step - accuracy: 0.7277 - loss: 0.5668 - val_accuracy: 0.7459 - val_loss: 0.5511
    Epoch 27/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 364ms/step - accuracy: 0.7313 - loss: 0.5583 - val_accuracy: 0.7855 - val_loss: 0.5364
    Epoch 28/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 358ms/step - accuracy: 0.7482 - loss: 0.5553 - val_accuracy: 0.5644 - val_loss: 0.9194
    Epoch 29/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 362ms/step - accuracy: 0.7581 - loss: 0.5405 - val_accuracy: 0.6799 - val_loss: 0.5810
    Epoch 30/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 366ms/step - accuracy: 0.7482 - loss: 0.5521 - val_accuracy: 0.7558 - val_loss: 0.5248
    Epoch 31/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.7440 - loss: 0.5506 - val_accuracy: 0.6931 - val_loss: 0.5598
    Epoch 32/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 361ms/step - accuracy: 0.7419 - loss: 0.5521 - val_accuracy: 0.7459 - val_loss: 0.5478
    Epoch 33/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 364ms/step - accuracy: 0.7412 - loss: 0.5516 - val_accuracy: 0.7657 - val_loss: 0.5349
    Epoch 34/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.7581 - loss: 0.5340 - val_accuracy: 0.7690 - val_loss: 0.5227
    Epoch 35/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 356ms/step - accuracy: 0.7412 - loss: 0.5438 - val_accuracy: 0.6865 - val_loss: 0.5866
    Epoch 36/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.7496 - loss: 0.5396 - val_accuracy: 0.7327 - val_loss: 0.5724
    Epoch 37/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 361ms/step - accuracy: 0.7659 - loss: 0.5357 - val_accuracy: 0.7261 - val_loss: 0.5627
    Epoch 38/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 366ms/step - accuracy: 0.7595 - loss: 0.5305 - val_accuracy: 0.7987 - val_loss: 0.5020
    Epoch 39/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 357ms/step - accuracy: 0.7553 - loss: 0.5287 - val_accuracy: 0.7096 - val_loss: 0.5819
    Epoch 40/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.7532 - loss: 0.5343 - val_accuracy: 0.7492 - val_loss: 0.5428
    Epoch 41/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 369ms/step - accuracy: 0.7702 - loss: 0.5265 - val_accuracy: 0.7459 - val_loss: 0.5394
    Epoch 42/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 362ms/step - accuracy: 0.7659 - loss: 0.5227 - val_accuracy: 0.7591 - val_loss: 0.5326
    Epoch 43/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 353ms/step - accuracy: 0.7666 - loss: 0.5216 - val_accuracy: 0.7129 - val_loss: 0.5460
    Epoch 44/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 361ms/step - accuracy: 0.7808 - loss: 0.5124 - val_accuracy: 0.7822 - val_loss: 0.5148
    Epoch 45/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 363ms/step - accuracy: 0.7730 - loss: 0.5184 - val_accuracy: 0.7657 - val_loss: 0.5417
    Epoch 46/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 356ms/step - accuracy: 0.7850 - loss: 0.5114 - val_accuracy: 0.7327 - val_loss: 0.5504
    Epoch 47/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 358ms/step - accuracy: 0.7822 - loss: 0.5099 - val_accuracy: 0.7129 - val_loss: 0.6126
    Epoch 48/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 353ms/step - accuracy: 0.7793 - loss: 0.5128 - val_accuracy: 0.7855 - val_loss: 0.5046
    Epoch 49/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 357ms/step - accuracy: 0.7702 - loss: 0.5186 - val_accuracy: 0.7360 - val_loss: 0.5444
    Epoch 50/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 359ms/step - accuracy: 0.7737 - loss: 0.5147 - val_accuracy: 0.6832 - val_loss: 0.5983
    Epoch 51/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 361ms/step - accuracy: 0.7864 - loss: 0.5175 - val_accuracy: 0.8053 - val_loss: 0.4987
    Epoch 52/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 355ms/step - accuracy: 0.7907 - loss: 0.4912 - val_accuracy: 0.6964 - val_loss: 0.6108
    Epoch 53/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 357ms/step - accuracy: 0.7864 - loss: 0.5161 - val_accuracy: 0.8350 - val_loss: 0.4770
    Epoch 54/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 355ms/step - accuracy: 0.7963 - loss: 0.4926 - val_accuracy: 0.8020 - val_loss: 0.4839
    Epoch 55/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 356ms/step - accuracy: 0.8041 - loss: 0.4904 - val_accuracy: 0.7558 - val_loss: 0.5652
    Epoch 56/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 372ms/step - accuracy: 0.8126 - loss: 0.4822 - val_accuracy: 0.8086 - val_loss: 0.4738
    Epoch 57/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 361ms/step - accuracy: 0.8020 - loss: 0.4840 - val_accuracy: 0.8383 - val_loss: 0.4698
    Epoch 58/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 366ms/step - accuracy: 0.8062 - loss: 0.4816 - val_accuracy: 0.8449 - val_loss: 0.4738
    Epoch 59/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 364ms/step - accuracy: 0.8168 - loss: 0.4738 - val_accuracy: 0.8251 - val_loss: 0.4764
    Epoch 60/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 378ms/step - accuracy: 0.8190 - loss: 0.4693 - val_accuracy: 0.7756 - val_loss: 0.5259
    Epoch 61/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 384ms/step - accuracy: 0.8168 - loss: 0.4730 - val_accuracy: 0.8383 - val_loss: 0.4640
    Epoch 62/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 378ms/step - accuracy: 0.8211 - loss: 0.4607 - val_accuracy: 0.7690 - val_loss: 0.5079
    Epoch 63/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 369ms/step - accuracy: 0.8239 - loss: 0.4648 - val_accuracy: 0.8251 - val_loss: 0.4776
    Epoch 64/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 371ms/step - accuracy: 0.8232 - loss: 0.4562 - val_accuracy: 0.7756 - val_loss: 0.5121
    Epoch 65/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 368ms/step - accuracy: 0.8126 - loss: 0.4660 - val_accuracy: 0.8218 - val_loss: 0.4495
    Epoch 66/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.8324 - loss: 0.4490 - val_accuracy: 0.8053 - val_loss: 0.5000
    Epoch 67/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 365ms/step - accuracy: 0.8154 - loss: 0.4706 - val_accuracy: 0.8218 - val_loss: 0.4486
    Epoch 68/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 359ms/step - accuracy: 0.8190 - loss: 0.4696 - val_accuracy: 0.8284 - val_loss: 0.4611
    Epoch 69/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 362ms/step - accuracy: 0.8380 - loss: 0.4463 - val_accuracy: 0.8383 - val_loss: 0.4401
    Epoch 70/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 359ms/step - accuracy: 0.8416 - loss: 0.4425 - val_accuracy: 0.8086 - val_loss: 0.4608
    Epoch 71/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 358ms/step - accuracy: 0.8437 - loss: 0.4410 - val_accuracy: 0.8218 - val_loss: 0.4569
    Epoch 72/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 354ms/step - accuracy: 0.8451 - loss: 0.4407 - val_accuracy: 0.8284 - val_loss: 0.4499
    Epoch 73/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 355ms/step - accuracy: 0.8331 - loss: 0.4386 - val_accuracy: 0.8185 - val_loss: 0.4641
    Epoch 74/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 356ms/step - accuracy: 0.8409 - loss: 0.4437 - val_accuracy: 0.8680 - val_loss: 0.4449
    Epoch 75/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 361ms/step - accuracy: 0.8402 - loss: 0.4347 - val_accuracy: 0.8284 - val_loss: 0.4611
    Epoch 76/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 358ms/step - accuracy: 0.8536 - loss: 0.4240 - val_accuracy: 0.8284 - val_loss: 0.4581
    Epoch 77/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 360ms/step - accuracy: 0.8437 - loss: 0.4323 - val_accuracy: 0.8383 - val_loss: 0.4585
    Epoch 78/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 358ms/step - accuracy: 0.8472 - loss: 0.4276 - val_accuracy: 0.8317 - val_loss: 0.4527
    Epoch 79/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 357ms/step - accuracy: 0.8571 - loss: 0.4216 - val_accuracy: 0.8350 - val_loss: 0.4634
    Epoch 80/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 356ms/step - accuracy: 0.8479 - loss: 0.4271 - val_accuracy: 0.8383 - val_loss: 0.4678
    Epoch 81/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 359ms/step - accuracy: 0.8515 - loss: 0.4212 - val_accuracy: 0.8449 - val_loss: 0.4402
    Epoch 82/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 357ms/step - accuracy: 0.8635 - loss: 0.4134 - val_accuracy: 0.8416 - val_loss: 0.4714
    Epoch 83/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 16s 357ms/step - accuracy: 0.8593 - loss: 0.4077 - val_accuracy: 0.8383 - val_loss: 0.4493
    Epoch 84/100
    45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 364ms/step - accuracy: 0.8642 - loss: 0.4119 - val_accuracy: 0.8449 - val_loss: 0.4509
    ============================================================
      TRAINING SUMMARY
    ============================================================
      Stopped at Epoch            : 84 / 100
      Best Epoch                  : 69
      Train Accuracy              : 83.80%
      Train Loss                  : 0.4463
      Val Accuracy                : 83.83%
      Val Loss                    : 0.4401
    ============================================================
:::
:::

::: {.cell .markdown}
### 5.6 Kurva Pelatihan {#56-kurva-pelatihan}

Kurva loss dan accuracy pada data latih dan data validasi ditampilkan
untuk mengamati proses konvergensi model serta mendeteksi indikasi
overfitting maupun underfitting selama pelatihan berlangsung.
:::

::: {.cell .code execution_count="29" execution="{\"iopub.execute_input\":\"2026-07-15T13:37:54.772966Z\",\"iopub.status.busy\":\"2026-07-15T13:37:54.772319Z\",\"iopub.status.idle\":\"2026-07-15T13:37:55.113226Z\",\"shell.execute_reply\":\"2026-07-15T13:37:55.112448Z\",\"shell.execute_reply.started\":\"2026-07-15T13:37:54.772934Z\"}" trusted="true"}
``` python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Model — Training Curves", fontsize=13, fontweight="bold")

axes[0].plot(history.history["loss"],     label="Train Loss", color="#4C72B0")
axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#DD8452", linestyle="--")
axes[0].axvline(x=best_epoch - 1, color="gray", linestyle=":", label=f"Best Epoch ({best_epoch})")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].spines[["top", "right"]].set_visible(False)

axes[1].plot(history.history["accuracy"],     label="Train Accuracy", color="#4C72B0")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#DD8452", linestyle="--")
axes[1].axvline(x=best_epoch - 1, color="gray", linestyle=":", label=f"Best Epoch ({best_epoch})")
axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/35f48c41cf6f5defd538debc48b3a4c76fd7abfb.png)
:::
:::

::: {.cell .markdown}
## 6. Evaluation {#6-evaluation}

Tahap Evaluation dilakukan untuk mengukur performa model pada data uji
(test set) yang tidak pernah dilihat model selama proses pelatihan
maupun validasi. Bobot yang digunakan adalah bobot terbaik yang telah
dipulihkan secara otomatis oleh EarlyStopping melalui parameter
`restore_best_weights=True`. Evaluasi mencakup perhitungan Confusion
Matrix, Accuracy, Precision, Recall (Sensitivity), Specificity,
F1-Score, dan ROC-AUC untuk memberikan gambaran performa model secara
komprehensif.
:::

::: {.cell .markdown}
### 6.1 Evaluasi pada Test Set {#61-evaluasi-pada-test-set}

Model dievaluasi menggunakan `test_gen` untuk memperoleh nilai loss dan
accuracy pada data uji. Hasil ini menjadi acuan awal sebelum dilakukan
analisis metrik yang lebih rinci pada subbagian berikutnya.
:::

::: {.cell .code execution_count="30" execution="{\"iopub.execute_input\":\"2026-07-15T13:37:55.114659Z\",\"iopub.status.busy\":\"2026-07-15T13:37:55.114249Z\",\"iopub.status.idle\":\"2026-07-15T13:37:58.494224Z\",\"shell.execute_reply\":\"2026-07-15T13:37:58.493202Z\",\"shell.execute_reply.started\":\"2026-07-15T13:37:55.114624Z\"}" trusted="true"}
``` python
test_loss, test_acc = model.evaluate(test_gen, verbose=0)

print("=" * 60)
print("  EVALUASI TEST SET")
print("=" * 60)
print(f"  {'Test Loss':<28}: {test_loss:.4f}")
print(f"  {'Test Accuracy':<28}: {test_acc * 100:.2f}%")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      EVALUASI TEST SET
    ============================================================
      Test Loss                   : 0.4721
      Test Accuracy               : 83.17%
    ============================================================
:::
:::

::: {.cell .markdown}
### 6.2 Prediksi pada Test Set {#62-prediksi-pada-test-set}

Seluruh citra dan label pada `test_gen` dikumpulkan melalui iterasi
batch, kemudian model menghasilkan probabilitas prediksi untuk setiap
citra. Label kelas ditentukan berdasarkan threshold 0.5: probabilitas di
atas atau sama dengan 0.5 diklasifikasikan sebagai tinea (kelas 1),
sedangkan di bawahnya diklasifikasikan sebagai eczema (kelas 0).
:::

::: {.cell .code execution_count="31" execution="{\"iopub.execute_input\":\"2026-07-15T13:37:58.495734Z\",\"iopub.status.busy\":\"2026-07-15T13:37:58.495212Z\",\"iopub.status.idle\":\"2026-07-15T13:38:02.383938Z\",\"shell.execute_reply\":\"2026-07-15T13:38:02.383200Z\",\"shell.execute_reply.started\":\"2026-07-15T13:37:58.495709Z\"}" trusted="true"}
``` python
X_test, y_true = [], []
for i in range(len(test_gen)):
    images, labels = test_gen[i]
    X_test.append(images)
    y_true.append(labels)

X_test = np.concatenate(X_test)
y_true = np.concatenate(y_true)

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("=" * 60)
print("  HASIL PREDIKSI PADA TEST SET")
print("=" * 60)
print(f"  {'Total Citra Diprediksi':<28}: {len(y_true)}")
print(f"  {'Threshold Klasifikasi':<28}: 0.5")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      HASIL PREDIKSI PADA TEST SET
    ============================================================
      Total Citra Diprediksi      : 303
      Threshold Klasifikasi       : 0.5
    ============================================================
:::
:::

::: {.cell .markdown}
### 6.3 Confusion Matrix {#63-confusion-matrix}

Confusion Matrix ditampilkan dalam dua bentuk berdampingan: raw count
untuk melihat jumlah prediksi secara langsung, dan normalized untuk
melihat proporsi prediksi benar dan salah pada setiap kelas secara
relatif terhadap jumlah data aktualnya.
:::

::: {.cell .code execution_count="32" execution="{\"iopub.execute_input\":\"2026-07-15T13:38:02.385422Z\",\"iopub.status.busy\":\"2026-07-15T13:38:02.384981Z\",\"iopub.status.idle\":\"2026-07-15T13:38:02.599392Z\",\"shell.execute_reply\":\"2026-07-15T13:38:02.598535Z\",\"shell.execute_reply.started\":\"2026-07-15T13:38:02.385393Z\"}" trusted="true"}
``` python
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")

for ax, data, fmt, title in zip(
    axes,
    [cm, cm_norm],
    ["d", ".2f"],
    ["Raw Count", "Normalized"]
):
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar=False
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print("=" * 60)
print("  KOMPONEN CONFUSION MATRIX")
print("=" * 60)
print(f"  {'True Positive (TP)':<28}: {tp}")
print(f"  {'True Negative (TN)':<28}: {tn}")
print(f"  {'False Positive (FP)':<28}: {fp}")
print(f"  {'False Negative (FN)':<28}: {fn}")
print("=" * 60)
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/08a95db0a6924a84b52ec8d5a19c06a385313ad6.png)
:::

::: {.output .stream .stdout}
    ============================================================
      KOMPONEN CONFUSION MATRIX
    ============================================================
      True Positive (TP)          : 120
      True Negative (TN)          : 132
      False Positive (FP)         : 23
      False Negative (FN)         : 28
    ============================================================
:::
:::

::: {.cell .markdown}
### 6.4 Classification Report {#64-classification-report}

Classification Report menyajikan Precision, Recall, dan F1-Score untuk
masing-masing kelas (eczema dan tinea), beserta rata-rata makro dan
rata-rata tertimbang (weighted average) yang memperhitungkan proporsi
jumlah data pada setiap kelas.
:::

::: {.cell .code execution_count="33" execution="{\"iopub.execute_input\":\"2026-07-15T13:38:02.600726Z\",\"iopub.status.busy\":\"2026-07-15T13:38:02.600370Z\",\"iopub.status.idle\":\"2026-07-15T13:38:02.615703Z\",\"shell.execute_reply\":\"2026-07-15T13:38:02.615004Z\",\"shell.execute_reply.started\":\"2026-07-15T13:38:02.600680Z\"}" trusted="true"}
``` python
print("Classification Report — Test Set")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
```

::: {.output .stream .stdout}
    Classification Report — Test Set
    ============================================================
                  precision    recall  f1-score   support

          eczema     0.8250    0.8516    0.8381       155
           tinea     0.8392    0.8108    0.8247       148

        accuracy                         0.8317       303
       macro avg     0.8321    0.8312    0.8314       303
    weighted avg     0.8319    0.8317    0.8316       303
:::
:::

::: {.cell .markdown}
### 6.5 Specificity {#65-specificity}

Specificity mengukur kemampuan model dalam mengenali kelas negatif
secara benar. Berbeda dengan Accuracy, Precision, Recall, dan F1-Score
yang telah tersedia melalui scikit-learn, Specificity dihitung secara
manual untuk setiap kelas berdasarkan komponen Confusion Matrix, dengan
memperlakukan masing-masing kelas secara bergantian sebagai kelas
positif (pendekatan one-vs-rest).
:::

::: {.cell .code execution_count="34" execution="{\"iopub.execute_input\":\"2026-07-15T13:38:02.617025Z\",\"iopub.status.busy\":\"2026-07-15T13:38:02.616824Z\",\"iopub.status.idle\":\"2026-07-15T13:38:02.633473Z\",\"shell.execute_reply\":\"2026-07-15T13:38:02.632682Z\",\"shell.execute_reply.started\":\"2026-07-15T13:38:02.617005Z\"}" trusted="true"}
``` python
def compute_specificity(cm):
    specificities = []
    n_classes = cm.shape[0]
    for i in range(n_classes):
        tn_i = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp_i = cm[:, i].sum() - cm[i, i]
        specificity_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
        specificities.append(specificity_i)
    return specificities

specificities        = compute_specificity(cm)
specificity_macro    = np.mean(specificities)

print("=" * 60)
print("  SPECIFICITY PER KELAS")
print("=" * 60)
for label, spec in zip(CLASS_NAMES, specificities):
    print(f"  {label.capitalize():<28}: {spec * 100:.2f}%")
print("-" * 60)
print(f"  {'Rata-rata (Macro)':<28}: {specificity_macro * 100:.2f}%")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      SPECIFICITY PER KELAS
    ============================================================
      Eczema                      : 81.08%
      Tinea                       : 85.16%
    ------------------------------------------------------------
      Rata-rata (Macro)           : 83.12%
    ============================================================
:::
:::

::: {.cell .markdown}
### 6.6 ROC Curve dan AUC {#66-roc-curve-dan-auc}

ROC Curve menggambarkan hubungan antara True Positive Rate dan False
Positive Rate pada berbagai nilai threshold, sedangkan AUC (Area Under
the Curve) merepresentasikan kemampuan model dalam membedakan kelas
eczema dan tinea secara keseluruhan, dengan nilai mendekati 1
menunjukkan performa yang semakin baik.
:::

::: {.cell .code execution_count="35" execution="{\"iopub.execute_input\":\"2026-07-15T13:38:02.634736Z\",\"iopub.status.busy\":\"2026-07-15T13:38:02.634420Z\",\"iopub.status.idle\":\"2026-07-15T13:38:02.794996Z\",\"shell.execute_reply\":\"2026-07-15T13:38:02.794142Z\",\"shell.execute_reply.started\":\"2026-07-15T13:38:02.634703Z\"}" trusted="true"}
``` python
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
auc_score            = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"Model (AUC = {auc_score:.4f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
ax.set_title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

print("=" * 60)
print(f"  AUC-ROC Score               : {auc_score:.4f} ({auc_score * 100:.2f}%)")
print("=" * 60)
```

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/cfd710bd02d79b6179c6724847f8992bafc7f5bb.png)
:::

::: {.output .stream .stdout}
    ============================================================
      AUC-ROC Score               : 0.8974 (89.74%)
    ============================================================
:::
:::

::: {.cell .markdown}
### 6.7 Ringkasan Metrik Evaluasi {#67-ringkasan-metrik-evaluasi}

Ringkasan berikut merangkum seluruh metrik evaluasi yang digunakan pada
penelitian ini, yaitu Accuracy, Precision, Recall (Sensitivity),
Specificity, F1-Score, dan ROC-AUC, untuk memberikan gambaran performa
model secara menyeluruh pada test set.
:::

::: {.cell .code execution_count="45" execution="{\"iopub.execute_input\":\"2026-07-15T13:46:17.636822Z\",\"iopub.status.busy\":\"2026-07-15T13:46:17.636339Z\",\"iopub.status.idle\":\"2026-07-15T13:46:17.670200Z\",\"shell.execute_reply\":\"2026-07-15T13:46:17.669497Z\",\"shell.execute_reply.started\":\"2026-07-15T13:46:17.636793Z\"}" trusted="true"}
``` python
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class    = recall_score(y_true, y_pred, average=None)
f1_per_class         = f1_score(y_true, y_pred, average=None)

# Rata-rata Macro (dibutuhkan untuk ringkasan & conclusion)
precision_macro = precision_score(y_true, y_pred, average="macro")
recall_macro    = recall_score(y_true, y_pred, average="macro")
f1_macro        = f1_score(y_true, y_pred, average="macro")

# Komponen Confusion Matrix per kelas (one-vs-rest)
tp_eczema, fn_eczema, fp_eczema, tn_eczema = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
tp_tinea,  fn_tinea,  fp_tinea,  tn_tinea  = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]

summary_table = pd.DataFrame([
    {
        "Kelas": "eczema",
        "TP": tp_eczema, "TN": tn_eczema, "FP": fp_eczema, "FN": fn_eczema,
        "Accuracy": test_acc,
        "Precision": precision_per_class[0],
        "Recall": recall_per_class[0],
        "Specificity": specificities[0],
        "F1-Score": f1_per_class[0],
        "ROC-AUC": auc_score,
    },
    {
        "Kelas": "tinea",
        "TP": tp_tinea, "TN": tn_tinea, "FP": fp_tinea, "FN": fn_tinea,
        "Accuracy": test_acc,
        "Precision": precision_per_class[1],
        "Recall": recall_per_class[1],
        "Specificity": specificities[1],
        "F1-Score": f1_per_class[1],
        "ROC-AUC": auc_score,
    },
])

display_table = summary_table.copy()
for col in ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]:
    display_table[col] = (display_table[col] * 100).round(2).astype(str) + "%"

print("=" * 110)
print("  RINGKASAN METRIK EVALUASI — TEST SET")
print("=" * 110)
print(display_table.to_string(index=False))
print("=" * 110)

print()
print("=" * 60)
print("  RATA-RATA MACRO — TEST SET")
print("=" * 60)
print(f"  {'Precision (Macro)':<22}: {precision_macro * 100:.2f}%")
print(f"  {'Recall (Macro)':<22}: {recall_macro * 100:.2f}%")
print(f"  {'F1-Score (Macro)':<22}: {f1_macro * 100:.2f}%")
print("=" * 60)
```

::: {.output .stream .stdout}
    ==============================================================================================================
      RINGKASAN METRIK EVALUASI — TEST SET
    ==============================================================================================================
     Kelas  TP  TN  FP  FN Accuracy Precision Recall Specificity F1-Score ROC-AUC
    eczema 132 120  28  23   83.17%     82.5% 85.16%      81.08%   83.81%  89.74%
     tinea 120 132  23  28   83.17%    83.92% 81.08%      85.16%   82.47%  89.74%
    ==============================================================================================================

    ============================================================
      RATA-RATA MACRO — TEST SET
    ============================================================
      Precision (Macro)     : 83.21%
      Recall (Macro)        : 83.12%
      F1-Score (Macro)      : 83.14%
    ============================================================
:::
:::

::: {.cell .markdown}
## 7. Inference {#7-inference}

Tahap Inference dilakukan untuk menguji kemampuan model dalam
mengklasifikasikan citra baru secara langsung, sebagai simulasi
penggunaan model pada skenario nyata. Sejumlah citra diambil secara acak
dari test set, kemudian diklasifikasikan oleh model beserta nilai
confidence-nya. Hasil prediksi dibandingkan dengan label sebenarnya
untuk melihat contoh kasus yang diprediksi benar maupun salah.
:::

::: {.cell .markdown}
### 7.1 Pemilihan Sampel Uji {#71-pemilihan-sampel-uji}

Sebanyak 6 citra diambil secara acak dari test set, terdiri atas 3 citra
eczema dan 3 citra tinea, untuk memastikan kedua kelas terwakili dalam
pengujian inference ini.
:::

::: {.cell .code execution_count="47" execution="{\"iopub.execute_input\":\"2026-07-15T13:46:26.758068Z\",\"iopub.status.busy\":\"2026-07-15T13:46:26.757280Z\",\"iopub.status.idle\":\"2026-07-15T13:46:26.764620Z\",\"shell.execute_reply\":\"2026-07-15T13:46:26.763853Z\",\"shell.execute_reply.started\":\"2026-07-15T13:46:26.758038Z\"}" trusted="true"}
``` python
random.seed(SEED)

eczema_idx = [i for i, label in enumerate(y_true) if label == 0]
tinea_idx  = [i for i, label in enumerate(y_true) if label == 1]

sample_idx    = random.sample(eczema_idx, 3) + random.sample(tinea_idx, 3)
sample_images = X_test[sample_idx]
sample_labels = y_true[sample_idx]

print("=" * 60)
print("  SAMPEL UJI INFERENCE")
print("=" * 60)
print(f"  {'Total Citra':<28}: 6 (3 Eczema, 3 Tinea)")
print("=" * 60)
```

::: {.output .stream .stdout}
    ============================================================
      SAMPEL UJI INFERENCE
    ============================================================
      Total Citra                 : 6 (3 Eczema, 3 Tinea)
    ============================================================
:::
:::

::: {.cell .markdown}
### 7.2 Prediksi dan Visualisasi {#72-prediksi-dan-visualisasi}

Model memprediksi keenam citra sampel dan menghasilkan confidence score
untuk setiap prediksi. Setiap citra ditampilkan beserta label
sebenarnya, label prediksi, dan confidence score. Border hijau
menandakan prediksi benar, sedangkan border merah menandakan prediksi
salah.
:::

::: {.cell .code execution_count="38" execution="{\"iopub.execute_input\":\"2026-07-15T13:38:02.836449Z\",\"iopub.status.busy\":\"2026-07-15T13:38:02.835984Z\",\"iopub.status.idle\":\"2026-07-15T13:38:03.753051Z\",\"shell.execute_reply\":\"2026-07-15T13:38:03.752044Z\",\"shell.execute_reply.started\":\"2026-07-15T13:38:02.836412Z\"}" trusted="true"}
``` python
sample_probs = model.predict(sample_images, verbose=0).flatten()
sample_preds = (sample_probs >= 0.5).astype(int)
sample_confs = np.where(sample_preds == 1, sample_probs, 1 - sample_probs)

correct = int(np.sum(sample_preds == sample_labels))

print("=" * 60)
print("  HASIL INFERENCE")
print("=" * 60)
print(f"  {'Prediksi Benar':<28}: {correct} / 6")
print(f"  {'Prediksi Salah':<28}: {6 - correct} / 6")
print(f"  {'Accuracy Sampel':<28}: {correct / 6 * 100:.2f}%")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("Inference — Sampel Test Set", fontsize=13, fontweight="bold")

for ax, img, true_label, pred_label, conf in zip(
    axes.flatten(), sample_images, sample_labels, sample_preds, sample_confs
):
    is_correct   = (true_label == pred_label)
    border_color = "#2ecc71" if is_correct else "#e74c3c"
    status_text  = "Correct" if is_correct else "Wrong"

    ax.imshow(img)
    ax.axis("off")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)

    ax.set_title(
        f"True  : {CLASS_NAMES[true_label].capitalize()}\n"
        f"Pred  : {CLASS_NAMES[pred_label].capitalize()}\n"
        f"Conf  : {conf * 100:.2f}%\n"
        f"{status_text}",
        fontsize=9,
        loc="left",
        color=border_color if not is_correct else "black"
    )

plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    ============================================================
      HASIL INFERENCE
    ============================================================
      Prediksi Benar              : 6 / 6
      Prediksi Salah              : 0 / 6
      Accuracy Sampel             : 100.00%
    ============================================================
:::

::: {.output .display_data}
![](vertopal_f2948109471549fe8eeb496d9e9b8276/54beb39ca737e455c5a66a0f820aeb33c4776bfc.png)
:::
:::

::: {.cell .markdown}
## 8. Model Export {#8-model-export}

Model final diekspor agar dapat digunakan pada tahap deployment. Format
yang dipilih disesuaikan dengan rencana implementasi berbasis Streamlit,
dengan mempertimbangkan kemungkinan penggunaan Flask atau FastAPI
sebagai alternatif framework backend. Kedua framework tersebut, termasuk
Streamlit, sama-sama berjalan pada lingkungan Python sehingga model
dapat dimuat langsung tanpa memerlukan konversi ke format khusus
perangkat mobile maupun browser.
:::

::: {.cell .markdown}
### 8.1 Ekspor Model {#81-ekspor-model}

Model diekspor ke dua format. Format Keras (.keras) digunakan sebagai
format utama karena ringkas, mudah dimuat kembali menggunakan
`tf.keras.models.load_model()`, dan menjadi format native
TensorFlow/Keras saat ini. Format SavedModel turut disertakan sebagai
alternatif karena bersifat lebih portabel dan umum digunakan pada
skenario serving di luar ekosistem Keras, sehingga tetap kompatibel
apabila deployment nantinya dilakukan melalui Flask atau FastAPI.
:::

::: {.cell .code execution_count="48" execution="{\"iopub.execute_input\":\"2026-07-15T13:46:36.017577Z\",\"iopub.status.busy\":\"2026-07-15T13:46:36.016984Z\",\"iopub.status.idle\":\"2026-07-15T13:46:37.290073Z\",\"shell.execute_reply\":\"2026-07-15T13:46:37.289333Z\",\"shell.execute_reply.started\":\"2026-07-15T13:46:36.017547Z\"}" trusted="true"}
``` python
EXPORT_DIR = "/kaggle/working/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Format Keras — digunakan untuk deployment Streamlit / Flask / FastAPI
keras_path = os.path.join(EXPORT_DIR, "model_final.keras")
model.save(keras_path)

# Format SavedModel — alternatif untuk skenario serving yang lebih portabel
saved_model_path = os.path.join(EXPORT_DIR, "saved_model")
model.export(saved_model_path)

print("=" * 60)
print("  MODEL EXPORT")
print("=" * 60)
print(f"  [OK] Keras       -> {keras_path}")
print(f"  [OK] SavedModel  -> {saved_model_path}/")
print("=" * 60)
```

::: {.output .stream .stdout}
    INFO:tensorflow:Assets written to: /kaggle/working/exports/saved_model/assets
:::

::: {.output .stream .stderr}
    INFO:tensorflow:Assets written to: /kaggle/working/exports/saved_model/assets
:::

::: {.output .stream .stdout}
    Saved artifact at '/kaggle/working/exports/saved_model'. The following endpoints are available:

    * Endpoint 'serve'
      args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='keras_tensor')
    Output Type:
      TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
    Captures:
      140380076970576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076973264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076974032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076974224: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076973456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076972880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076975184: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076975760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076976144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076976912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076975952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076975568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076979216: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076980752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076981520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076981136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076981712: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076980944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076978064: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076978640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140380076982096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456281360: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456281744: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456282128: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456279632: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456280976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456282704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456284432: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456285200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456285584: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456284816: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456284240: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456283088: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456283664: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456285776: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456287504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456287696: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456288080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456288464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456286544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456286928: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456288656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456290192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456285392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456291344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456290000: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140377456292496: TensorSpec(shape=(), dtype=tf.resource, name=None)
    ============================================================
      MODEL EXPORT
    ============================================================
      [OK] Keras       -> /kaggle/working/exports/model_final.keras
      [OK] SavedModel  -> /kaggle/working/exports/saved_model/
    ============================================================
:::
:::

::: {.cell .markdown}
### 8.2 Verifikasi Model Hasil Ekspor {#82-verifikasi-model-hasil-ekspor}

Model yang telah diekspor dimuat kembali dan diuji menggunakan sampel
dari test set untuk memastikan hasil prediksinya konsisten dengan model
sebelum diekspor. Verifikasi ini penting untuk memastikan tidak ada
perubahan bobot maupun arsitektur selama proses penyimpanan.
:::

::: {.cell .code execution_count="49" execution="{\"iopub.execute_input\":\"2026-07-15T13:46:41.211576Z\",\"iopub.status.busy\":\"2026-07-15T13:46:41.210786Z\",\"iopub.status.idle\":\"2026-07-15T13:46:42.205848Z\",\"shell.execute_reply\":\"2026-07-15T13:46:42.205159Z\",\"shell.execute_reply.started\":\"2026-07-15T13:46:41.211546Z\"}" trusted="true"}
``` python
loaded_model = tf.keras.models.load_model(keras_path)

original_probs = model.predict(sample_images, verbose=0).flatten()
loaded_probs   = loaded_model.predict(sample_images, verbose=0).flatten()

is_identical = np.allclose(original_probs, loaded_probs, atol=1e-6)

print("=" * 60)
print("  VERIFIKASI MODEL HASIL EKSPOR")
print("=" * 60)
print(f"  {'Prediksi Model Asli':<28}: {np.round(original_probs, 4)}")
print(f"  {'Prediksi Model Dimuat':<28}: {np.round(loaded_probs, 4)}")
print("-" * 60)
print(f"  {'Hasil Identik':<28}: {'Ya' if is_identical else 'Tidak'}")
print("=" * 60)
```

::: {.output .stream .stdout}
    WARNING:tensorflow:5 out of the last 17 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7fac713c3380> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:5 out of the last 17 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7fac713c3380> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
:::

::: {.output .stream .stdout}
    ============================================================
      VERIFIKASI MODEL HASIL EKSPOR
    ============================================================
      Prediksi Model Asli         : [0.0354 0.3205 0.1523 0.5653 0.9494 0.9123]
      Prediksi Model Dimuat       : [0.0354 0.3205 0.1523 0.5653 0.9494 0.9123]
    ------------------------------------------------------------
      Hasil Identik               : Ya
    ============================================================
:::
:::

::: {.cell .markdown}
## 9. Conclusion {#9-conclusion}

Bagian ini merangkum keseluruhan proses pengembangan model, mulai dari
karakteristik data, arsitektur dan strategi pelatihan yang digunakan,
hingga performa akhir model pada test set. Ringkasan disusun secara
otomatis berdasarkan hasil eksekusi notebook sehingga tetap konsisten
apabila dilakukan pelatihan ulang (*run all*) dengan hasil performa yang
berbeda.
:::

::: {.cell .code execution_count="50" execution="{\"iopub.execute_input\":\"2026-07-15T13:46:49.722517Z\",\"iopub.status.busy\":\"2026-07-15T13:46:49.722039Z\",\"iopub.status.idle\":\"2026-07-15T13:46:49.734670Z\",\"shell.execute_reply\":\"2026-07-15T13:46:49.733680Z\",\"shell.execute_reply.started\":\"2026-07-15T13:46:49.722449Z\"}" trusted="true"}
``` python
import textwrap

WRAP_WIDTH = 70

def wrap_paragraph(text):
    return textwrap.fill(" ".join(text.split()), width=WRAP_WIDTH)

# Interpretasi kondisi EarlyStopping
early_stop_triggered = stopped_epoch < EPOCHS
if early_stop_triggered:
    early_stop_note = (
        f"EarlyStopping aktif menghentikan pelatihan lebih awal pada "
        f"epoch {stopped_epoch} dari {EPOCHS} epoch maksimum, setelah "
        f"val_loss tidak menunjukkan perbaikan selama "
        f"{callbacks[0].patience} epoch berturut-turut."
    )
else:
    early_stop_note = (
        f"Pelatihan berjalan hingga batas maksimum {EPOCHS} epoch "
        f"tanpa terpicu EarlyStopping, dengan performa terbaik "
        f"diperoleh pada epoch {best_epoch}."
    )

# Interpretasi gap overfitting
acc_gap = abs(train_acc - val_acc) * 100
if acc_gap < 5:
    gap_note = f"gap yang kecil ({acc_gap:.2f}%), menandakan model tidak mengalami overfitting signifikan"
elif acc_gap < 10:
    gap_note = f"gap yang moderat ({acc_gap:.2f}%), menandakan sedikit indikasi overfitting"
else:
    gap_note = f"gap yang cukup besar ({acc_gap:.2f}%), menandakan model berpotensi mengalami overfitting"

# Interpretasi pencapaian target performa
if test_acc >= 0.90:
    target_note = "telah mencapai target performa yang diharapkan (di atas 90%)"
elif test_acc >= 0.85:
    target_note = "mendekati target performa yang diharapkan (90-95%), namun belum tercapai sepenuhnya"
else:
    target_note = "masih berada di bawah target performa yang diharapkan (90-95%), sehingga optimasi lanjutan diperlukan"

# Identifikasi kelas dengan Recall terendah (paling berisiko false negative)
weakest_recall_idx   = int(np.argmin(recall_per_class))
weakest_recall_class = CLASS_NAMES[weakest_recall_idx]
weakest_recall_value = recall_per_class[weakest_recall_idx] * 100

# Susun paragraf
paragraph_1 = wrap_paragraph(f"""
Penelitian ini membangun model Convolutional Neural Network (CNN)
yang dilatih secara training from scratch untuk mengklasifikasikan
penyakit kulit eczema dan tinea, menggunakan dataset sebanyak
{final_total} citra setelah melalui tahap Data Cleaning
(Eczema = {final_eczema}, Tinea = {final_tinea}).
""")

paragraph_2 = wrap_paragraph(f"""
Model dikembangkan berdasarkan arsitektur Residual CNN dengan skip
connection dan SeparableConv2D, dilatih menggunakan strategi Cosine
Annealing Learning Rate Schedule dengan fase warmup serta Label
Smoothing sebesar {LABEL_SMOOTHING} pada fungsi loss. {early_stop_note}
""")

paragraph_3 = wrap_paragraph(f"""
Pada Train dan Validation set, model mencapai Train Accuracy sebesar
{train_acc * 100:.2f}% dan Validation Accuracy sebesar
{val_acc * 100:.2f}%, dengan {gap_note} antara keduanya.
""")

paragraph_4 = "Pada Test Set, model menghasilkan performa sebagai berikut:"

metrics_block = (
    f"  - {'Accuracy':<13}: {test_acc * 100:.2f}%\n"
    f"  - {'Precision':<13}: {precision_macro * 100:.2f}%\n"
    f"  - {'Recall':<13}: {recall_macro * 100:.2f}%\n"
    f"  - {'Specificity':<13}: {specificity_macro * 100:.2f}%\n"
    f"  - {'F1-Score':<13}: {f1_macro * 100:.2f}%\n"
    f"  - {'ROC-AUC':<13}: {auc_score * 100:.2f}%"
)

paragraph_5 = wrap_paragraph(f"""
Berdasarkan hasil tersebut, model {target_note}. Ditinjau per kelas,
Recall terendah diperoleh pada kelas {weakest_recall_class}
({weakest_recall_value:.2f}%), menunjukkan kelas tersebut memiliki
risiko false negative relatif lebih tinggi dibandingkan kelas
lainnya dan dapat menjadi perhatian pada tahap optimasi selanjutnya.
""")

paragraph_6 = wrap_paragraph("""
Model final telah diekspor ke format Keras (.keras) dan SavedModel
untuk mendukung tahap deployment berbasis Streamlit, dengan
kemungkinan implementasi alternatif menggunakan Flask maupun FastAPI.
""")

# Cetak hasil
print("=" * 70)
print("  CONCLUSION")
print("=" * 70)
print()
print(paragraph_1)
print()
print(paragraph_2)
print()
print(paragraph_3)
print()
print(paragraph_4)
print(metrics_block)
print()
print(paragraph_5)
print()
print(paragraph_6)
print()
print("=" * 70)
```

::: {.output .stream .stdout}
    ======================================================================
      CONCLUSION
    ======================================================================

    Penelitian ini membangun model Convolutional Neural Network (CNN) yang
    dilatih secara training from scratch untuk mengklasifikasikan penyakit
    kulit eczema dan tinea, menggunakan dataset sebanyak 2020 citra
    setelah melalui tahap Data Cleaning (Eczema = 1037, Tinea = 983).

    Model dikembangkan berdasarkan arsitektur Residual CNN dengan skip
    connection dan SeparableConv2D, dilatih menggunakan strategi Cosine
    Annealing Learning Rate Schedule dengan fase warmup serta Label
    Smoothing sebesar 0.1 pada fungsi loss. EarlyStopping aktif
    menghentikan pelatihan lebih awal pada epoch 84 dari 100 epoch
    maksimum, setelah val_loss tidak menunjukkan perbaikan selama 15 epoch
    berturut-turut.

    Pada Train dan Validation set, model mencapai Train Accuracy sebesar
    83.80% dan Validation Accuracy sebesar 83.83%, dengan gap yang kecil
    (0.02%), menandakan model tidak mengalami overfitting signifikan
    antara keduanya.

    Pada Test Set, model menghasilkan performa sebagai berikut:
      - Accuracy     : 83.17%
      - Precision    : 83.21%
      - Recall       : 83.12%
      - Specificity  : 83.12%
      - F1-Score     : 83.14%
      - ROC-AUC      : 89.74%

    Berdasarkan hasil tersebut, model masih berada di bawah target
    performa yang diharapkan (90-95%), sehingga optimasi lanjutan
    diperlukan. Ditinjau per kelas, Recall terendah diperoleh pada kelas
    tinea (81.08%), menunjukkan kelas tersebut memiliki risiko false
    negative relatif lebih tinggi dibandingkan kelas lainnya dan dapat
    menjadi perhatian pada tahap optimasi selanjutnya.

    Model final telah diekspor ke format Keras (.keras) dan SavedModel
    untuk mendukung tahap deployment berbasis Streamlit, dengan
    kemungkinan implementasi alternatif menggunakan Flask maupun FastAPI.

    ======================================================================
:::
:::

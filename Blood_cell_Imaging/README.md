# 🩸 Blood Cell Classification

A deep learning project that classifies white blood cell images into 4 categories using a Custom CNN and MobileNetV2 with Transfer Learning.

| Class | Description |
|-------|-------------|
| EOSINOPHIL | Immune response / allergies |
| LYMPHOCYTE | Antibody production / adaptive immunity |
| MONOCYTE | Inflammation / pathogen detection |
| NEUTROPHIL | First-line defense against bacteria |

---

## 📁 Project Files

```
BLOOD_CELL_IMAGING/
├── blood_cell_remixed.ipynb          # Main notebook — full pipeline
├── best_blood_cell_model.keras       # Saved Custom CNN (best weights)
├── blood_cell_mobilenetv2_final.keras# Saved MobileNetV2 (fine-tuned)
└── classification_report.csv         # Per-class evaluation metrics
```

---

## 🗺️ Pipeline Overview

```
1. Dataset loading (Kaggle — paultimothymooney/blood-cells)
2. EDA + 6 visualizations
3. Preprocessing & Data Augmentation
4. Stratified 70/15/15 train/val/test split (TRAIN+TEST pooled)
5. Model 1 — Custom CNN (trained from scratch)
6. Model 2 — MobileNetV2 (Phase 1: frozen base → Phase 2: fine-tuning)
7. Evaluation — Confusion Matrix, Classification Report, F1/Precision/Recall
8. Error Analysis
```

> **Note on the data split:** The Kaggle dataset's predefined `TRAIN/` and `TEST/` folders are not from the same distribution, which causes inflated validation scores (~90%) but low test scores (~56%) when evaluated directly against `TEST/`. The notebook pools both folders and creates its own stratified split to fix this.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/blood-cell-imaging.git
cd blood-cell-imaging
```

### 2. Install dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas pillow kagglehub
```

### 3. Configure Kaggle credentials

```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Run the notebook

```bash
jupyter notebook blood_cell_remixed.ipynb
```

Run all cells top to bottom. The dataset will be downloaded automatically via `kagglehub`.

---

## ♻️ Load Pre-trained Models (skip training)

```python
import tensorflow as tf

# Custom CNN
cnn = tf.keras.models.load_model("best_blood_cell_model.keras")

# MobileNetV2
mobilenet = tf.keras.models.load_model("blood_cell_mobilenetv2_final.keras")

# Predict
import numpy as np
from PIL import Image

img = np.array(Image.open("your_image.jpg").resize((160, 160))) / 255.0
img = np.expand_dims(img, axis=0)

classes = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
pred = mobilenet.predict(img)
print("Predicted:", classes[np.argmax(pred)])
```

---

## 📊 Results

See `classification_report.csv` for the full per-class breakdown.

| Model | Test Accuracy |
|-------|--------------|
| MobileNetV2 (fine-tuned) | see report |

MobileNetV2 with fine-tuning consistently outperforms the Custom CNN thanks to ImageNet pre-training.

---

## 🧰 Tech Stack

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- matplotlib / seaborn
- Pillow
- kagglehub

---

## 📄 Dataset

[Blood Cell Images — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)  
~12,500 augmented JPEG images across 4 white blood cell types.

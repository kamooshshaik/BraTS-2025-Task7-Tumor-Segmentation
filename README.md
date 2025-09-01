# BraTS 2025 — Task 7: Brain Tumor Segmentation (MRI)

> End‑to‑end deep learning pipeline for brain tumor segmentation on the BraTS dataset (Task 7).  
> Includes data prep, model training, evaluation, and ready-to-share artifacts (weights + notebooks).

## 🔍 Project at a glance 
- **Goal:** Segment tumor sub‑regions from multimodal MRI (FLAIR, T1, T1ce, T2).  
- **Approach:** U‑Net style CNN with data augmentation and Dice‑optimized training.  
- **Deliverables:** Reproducible notebook(s), trained weights (`.h5`), and evaluation scripts.  
- **Impact:** Automates radiology workflows; improves reproducibility of medical AI experiments.

---

## 📂 Repository Structure
```
.
├── brats2025_task7.ipynb         # Main training/inference notebook
├── best_model.weights.h5         # Best checkpoint (Git LFS recommended)
├── weights_epoch_045.weights.h5  # Example epoch checkpoint
├── brats_train_val_data.zip      # (Optional) Sample data pack (do not commit large zips)
├── README.md
├── requirements.txt
└── LICENSE
```
> Tip: Keep large files out of git history. Use **Git LFS** for `.h5` or upload to release assets/Drive.

---

## 🧠 Methodology
- **Backbone:** U‑Net (encoder–decoder with skip connections).  
- **Input:** Multimodal MRI volumes/slices (BraTS convention).  
- **Loss:** Dice (± focal/CE blend) to handle class imbalance.  
- **Metrics:** Dice score per sub‑region (WT/TC/ET), precision/recall, Hausdorff (optional).  
- **Augmentation:** Random flip/rotate, intensity shift, elastic deformation (as available).  
- **Training:** Early stopping + best‑checkpoint saving.

> This repository includes two Jupyter notebooks so recruiters can review the full workflow without setting up a package.

---

## 🚀 Quickstart

### 1) Clone and set up
```bash
git clone https://github.com/<your-username>/BraTS-2025-Task7-Tumor-Segmentation.git
cd BraTS-2025-Task7-Tumor-Segmentation
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data
- Use official **BraTS** dataset (register on the organizer's site) or your curated subset.  
- Put data under `data/` (ignored by git). For quick tests you can extract `brats_train_val_data.zip` locally.

```
data/
 ├── train/
 ├── val/
 └── test/
```

### 3) Run
- Open `brats2025_task7.ipynb` and run cells end‑to‑end (training + evaluation + inference).  
- Trained weights will be saved as `.h5` under `models/` or project root (as configured in the notebook).

---

## 📊 Results (template)
| Sub‑region | Dice (↑) | Precision (↑) | Recall (↑) |
|-----------:|:--------:|:-------------:|:----------:|
| Whole Tumor (WT) | `0.90±0.02` | `0.91` | `0.89` |
| Tumor Core (TC)  | `0.86±0.03` | `0.87` | `0.85` |
| Enhancing Tumor (ET) | `0.79±0.05` | `0.81` | `0.78` |

> Replace the above with your actual numbers. If you have sample predictions, add a `media/` folder with before/after images.

---

## 🧩 Key Files
- **`brats2025_task7.ipynb`** — clean, reproducible pipeline (data prep → training → eval → inference).  
- **`brats-task7.ipynb`** — ablations/experiments.  
- **`best_model.weights.h5`** — best performing checkpoint (use **Git LFS**).

---

## 🛡️ Ethics & Compliance
- Medical data must be used under the dataset license/consent.  
- Models are **research‑only**; not a clinical device. Validate thoroughly before any clinical use.

---


**Built a U‑Net based MRI pipeline for BraTS brain‑tumor segmentation (Task 7), delivered trained weights and reproducible notebooks, and reported Dice‑optimized results across tumor sub‑regions.**

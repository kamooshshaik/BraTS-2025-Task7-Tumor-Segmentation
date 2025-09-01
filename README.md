# 🧠 BraTS Task 7: Brain Tumor Segmentation using Deep Learning Algorithms

This repository contains the implementation of an **Enhanced 3D U-Net with Attention Gates** for **Brain Tumor Segmentation** on the **BraTS-GoAT 2025 dataset**.  
The pipeline includes **preprocessing, model training, validation, testing, and visualization** for clinical-grade segmentation.

---

## 📌 Table of Contents
1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Usage](#usage)
10. [References](#references)

---

## ⚙️ Requirements

Install required dependencies:

```bash
pip install numpy pandas tensorflow nibabel scikit-learn matplotlib seaborn tqdm
```

Main libraries:
- **TensorFlow / Keras** → Deep learning framework  
- **Nibabel** → Read MRI `.nii` files  
- **NumPy / Pandas** → Data handling  
- **Matplotlib / Seaborn** → Visualizations  
- **tqdm** → Progress bars  

> ⚡ GPU (NVIDIA T4 / V100 / A100) recommended. Mixed precision training is enabled.

---

## 📂 Dataset

We used the **BraTS-GoAT 2025 dataset**.

- **Modalities**: T1, T1c, T2, FLAIR  
- **Classes**: Background, Edema, Enhancing Tumor  
- **Usable cases**: ~1,100  
  - Training: 979  
  - Validation: 110  
- **Patch size**: `(48, 48, 48, 4)` input, `(48, 48, 48)` labels  

👉 Download dataset from [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2025/).

---

## 🧹 Preprocessing

Steps applied before training:

1. Load MRI volumes (`.nii`) using Nibabel  
2. Normalize intensities to a fixed range  
3. Handle corrupted/missing scans  
4. Extract **patches** of size `(48, 48, 48, 4)`  
5. Generate training/validation splits  

Example preprocessing script:

```bash
python preprocess.py --input /path/to/raw_data --output /path/to/processed_data
```

---

## 🏗️ Model Architecture

We implemented an **Enhanced 3D U-Net with Attention Gates**.

- **Input**: `(48, 48, 48, 4)`  
- **Output classes**: 3 (background, edema, enhancing)  
- **Parameters**: ~22.6M trainable  
- **Loss**: Hybrid (Dice + Cross-Entropy, α = 0.5)  
- **Metrics**: Dice coefficient, Sparse Categorical Accuracy  

---

## 🚀 Training

Training configuration:
- **Optimizer**: Adam (`lr=1e-4`)  
- **Batch size**: 2  
- **Epochs**: 100 (with checkpoints)  
- **Mixed precision**: Enabled  
- **Data augmentation**: random flips, rotations, intensity shifts  

Run training:

```bash
python train.py --data /path/to/processed_data --epochs 100 --batch 2
```

Checkpoints will be saved in `/model_checkpoints/`.

---

## 📊 Evaluation

Load best model weights and evaluate:

```python
from model import build_unet
import tensorflow as tf

model = build_unet()
model.load_weights("model_checkpoints/best_model.weights.h5")

results = model.evaluate(val_dataset)
print("Validation Results:", results)
```

Validation performance:
- **Accuracy**: 99.8%  
- **Dice coefficient**: ~0.78 overall  
- **Best-case tumor Dice**: 0.92  
- **Prediction time**: 0.094s per case  

---

## 🏆 Results

### Random Test Cases
- **Case BraTS-GoAT-01845** → Tumor Dice: **0.9202**  
- **Case BraTS-GoAT-02357** → Tumor Dice: **0.8862**  
- **Case BraTS-GoAT-00084** → Tumor Dice: **0.3528** (challenging case)  

### Summary Statistics
- **Accuracy**: Mean 0.9959 ± 0.0046  
- **Dice Scores**:  
  - Background: 0.9990 ± 0.0008  
  - Edema: 0.7242 ± 0.2644  
  - Enhancing: 0.7839 ± 0.1560  
  - Mean Tumor: 0.7541 ± 0.2061  
- **Mean prediction time**: 0.094s  

---

## 🔮 Future Work

- Improve segmentation for **low-contrast tumors**  
- Integrate **transformer-based models (TransUNet, SwinUNETR)**  
- Deploy with **ONNX/TensorRT** for faster inference  
- Explore **semi-supervised learning** with unlabeled MRI data  

---

## ▶️ Usage

1. **Clone repo**  
   ```bash
   git clone https://github.com/kamooshshaik/BraTS-2025-Task7-Tumor-Segmentation.git
   cd BraTS-2025-Task7-Tumor-Segmentation
   ```

2. **Preprocess dataset**  
   ```bash
   python preprocess.py --input /dataset --output /processed_data
   ```

3. **Train model**  
   ```bash
   python train.py --data /processed_data --epochs 100 --batch 2
   ```

4. **Evaluate model**  
   ```bash
   python evaluate.py --weights model_checkpoints/best_model.weights.h5 --data /processed_data
   ```

5. **Visualize predictions**  
   ```bash
   python visualize.py --weights model_checkpoints/best_model.weights.h5 --sample BraTS-GoAT-01845
   ```

---

## 📖 References
- [BraTS Challenge 2025](https://www.med.upenn.edu/cbica/brats2025/)  
- Olaf Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*  
- Isensee et al., *nnU-Net: A Self-adapting Framework for Biomedical Image Segmentation*  

---

## 👨‍💻 Author
Developed by **Shaik Kamoosh Baba**  
M.Tech – Civil Engineering (Geoinformatics), IIT Kanpur  

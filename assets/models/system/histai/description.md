# Hibou Foundation Models Descriptions

## 1. Model Overview

Hibou-B is a foundational vision transformer developed for digital pathology. The model is trained using self-supervised learning (DINOv2).

---

## 2. Technical Details

### 2.1 Architectures
- **Hibou-B**: Based on **ViT-B/14** with registers.

### 3.2 Training Dataset

- **Total WSIs**: 1.3 million  
  - **H&E-stained slides**: 936,441  
  - **Non-H&E slides**: 202,464  
  - **Cytology slides**: 2,676  
- **Sources**: 306,400 unique cases, including human and veterinary pathology.

### 3.3 Data Preprocessing

- WSIs split into **patches**, removing background via **Otsu thresholding.**
- **Hibou-B**: Trained on **512M clean patches.**

### 3.4 Data Augmentations
- Random rotation, horizontal & vertical flips.
- **RandStainNA** for stain normalization.
- **Color jittering**.

### 3.5 Training Infrastructure

| Model | GPU Setup | Iterations | Batch Size |
|--------|-------------|-----------|------------|
| Hibou-B | 8x A100-80G GPUs | 500K | 1024 |

---

## 4. Testing & Validation

### 4.1 Patch-Level Evaluation

| Dataset | Size | Hibou-B |
|---------|------|---------|
| CRC-100K | 107,180 | 0.955 |
| PCAM | 327,680 | 0.946 |
| MHIST | 3,152 | 0.812 |
| MSI-CRC | 193,312 | 0.779 |
| MSI-STAD | 218,578 | 0.797 |
| TIL-DET | 304,097 | 0.942 |

### 4.2 Slide-Level Evaluation

| Dataset | Size | Hibou-B (AUC) |
|---------|------|--------------|
| BRCA | 963 | 0.929 |
| NSCLC | 973 | 0.952 |
| RCC | 927 | 0.993 |

---

## 5. License

| Model | License |
|--------|-----------|
| Hibou-B | Apache 2.0 |

---
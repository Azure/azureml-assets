<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `evaluation.md` is highly recommended, but not required. It captures information about the performance of the model. We highly recommend including this section as this information is often used to decide what model to use. -->

We benchmarked MedImageParse 3D against task-specific nnU-Net models on AMOS22 CT and MRI datasets. Note that we trained a single model to solve all different tasks solely via text prompting, e.g. "gallbladder in abdomen MRI", while nnU-Net was trained as multiple expert models for each individual object in each modality. Therefore, we made this comparison of one single model v.s. 27 task-specific models.


**CT**

| Dice score (%)   | aorta | bladder | duodenum | esophagus | gallbladder | left adrenal gland | left kidney | liver | pancreas | IVC   | right adrenal gland | right kidney | spleen | stomach | Average |
| ---------------- | ----- | ------- | -------- | --------- | ----------- | ------------------ | ----------- | ----- | -------- | ----- | ------------------- | ------------ | ------ | ------- | ------- |
| MedImageParse 3D | 95.27 | 90.17   | 83.27    | 87.11     | 85.96       | 79.48              | 96.39       | 97.71 | 88.42    | 92.02 | 79.39               | 96.88        | 96.91  | 91.49   | 90.00   |
| nnU-Net          | 95.20 | 87.52   | 80.72    | 87.31     | 83.06       | 78.06              | 95.39       | 96.09 | 86.57    | 90.38 | 78.24               | 93.19        | 96.91  | 89.79   | 88.35   |
| SegVol           | 92.07 | 88.03   | 72.49    | 64.47     | 79.05       | 76.31              | 94.58       | 96.24 | 80.97    | 83.65 | 71.07               | 92.92        | 94.03  | 88.82   | 83.75   |

**MRI**

| Dice score (%)   | aorta | duodenum | esophagus | gallbladder | left adrenal gland | left kidney | liver | pancreas | IVC   | right adrenal gland | right kidney | spleen | stomach | Average |
| ---------------- | ----- | -------- | --------- | ----------- | ------------------ | ----------- | ----- | -------- | ----- | ------------------- | ------------ | ------ | ------- | ------- |
| MedImageParse 3D | 95.73 | 76.03    | 81.38     | 66.58       | 63.35              | 96.92       | 97.65 | 88.70    | 87.26 | 68.14               | 96.69        | 96.88  | 88.93   | 84.94   |
| nnU-Net          | 95.64 | 66.78    | 73.62     | 66.32       | 57.15              | 95.82       | 97.25 | 79.29    | 90.66 | 53.29               |              |        |         |         |

---

We evaluated MedImageParse 3D on the CVPR 2025 Foundation Models for Text-guided 3D Biomedical Image Segmentation open-challenge validation set, which includes both semantic and instance segmentation tasks. For semantic segmentation, we report Dice Similarity Coefficient (DSC) for region overlap and Normalized Surface Distance (NSD) for boundary accuracy. For instance segmentation, we report the F1 score at an IoU threshold of 0.5 and DSC for true-positive instances.


**CT**

| Method           | DSC (Semantic) | NSD (Semantic) | F1 (Instance) | DSC TP (Instance) |
| ---------------- | -------------- | -------------- | ------------- | ----------------- |
| CAT              | 0.7211         | 0.7227         | 0.2993        | 0.3717            |
| SAT              | 0.6780         | 0.6726         | 0.2517        | 0.3954            |
| MedImageParse 3D | 0.8512         | 0.8965         | 0.5119        | 0.6749            |

**MRI**

| Method           | DSC (Semantic) | NSD (Semantic) | F1 (Instance) | DSC TP (Instance) |
| ---------------- | -------------- | -------------- | ------------- | ----------------- |
| CAT              | 0.5415         | 0.6193         | 0.1375        | 0.2813            |
| SAT              | 0.5610         | 0.6669         | 0.1228        | 0.2728            |
| MedImageParse 3D | 0.7396         | 0.8664         | 0.5317        | 0.7053            |

**Microscopy**

| Method           | DSC (Semantic) | NSD (Semantic) | F1 (Instance) | DSC TP (Instance) |
| ---------------- | -------------- | -------------- | ------------- | ----------------- |
| CAT              | –             | –             | 0.0313        | 0.3628            |
| SAT              | –             | –             | 0.2006        | 0.4243            |
| MedImageParse 3D | –             | –             | 0.1939        | 0.6552            |

**PET**

| Method           | DSC (Semantic) | NSD (Semantic) | F1 (Instance) | DSC TP (Instance) |
| ---------------- | -------------- | -------------- | ------------- | ----------------- |
| CAT              | –             | –             | 0.1098        | 0.2779            |
| SAT              | –             | –             | 0.4200        | 0.7863            |
| MedImageParse 3D | –             | –             | 0.3132        | 0.7185            |

**Ultrasound**

| Method           | DSC (Semantic) | NSD (Semantic) | F1 (Instance) | DSC TP (Instance) |
| ---------------- | -------------- | -------------- | ------------- | ----------------- |
| CAT              | 0.8594         | 0.8360         | –            | –                |
| SAT              | 0.8558         | 0.7924         | –            | –                |
| MedImageParse 3D | 0.9050         | 0.9135         | –            | –                |

*Note: “–” indicates that the metric is not applicable for that modality/method.*

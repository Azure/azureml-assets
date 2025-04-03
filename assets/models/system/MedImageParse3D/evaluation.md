<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `evaluation.md` is highly recommended, but not required. It captures information about the performance of the model. We highly recommend including this section as this information is often used to decide what model to use. -->

We benchmarked MedImageParse 3D against task-specific nnU-Net models on AMOS22 CT and MRI datasets. Note that we trained a single model to solve all different tasks solely via text prompting, e.g. "gallbladder in abdomen MRI", while nnU-Net was trained as multiple expert models for each individual object in each modality. Therefore, we made this comparison of one single model v.s. 27 task-specific models.

**CT**
| Dice score (%)           | aorta  | bladder | duodenum | esophagus | gallbladder | left adrenal gland | left kidney | liver  | pancreas | IVC | right adrenal gland | right kidney | spleen | stomach | Average   |
|-----------------|--------|---------|----------|-----------|-------------|---------------------|-------------|--------|----------|----------|---------------------|-------------|--------|---------|--------|
| MedImageParse 3D  | 95.27 | 90.17  | 83.27   | 87.11    | 85.96      | 79.48              | 96.39      | 97.71 | 88.42   | 92.02   | 79.39              | 96.88      | 96.91 | 91.49  | 90.00 |
| nnU-Net        | 95.20 | 87.52  | 80.72   | 87.31    | 83.06      | 78.06              | 95.39      | 96.09 | 86.57   | 90.38   | 78.24              | 93.19      | 96.91 | 89.79  | 88.35 |
| SegVol      | 92.07  | 88.03   | 72.49    | 64.47    | 79.05       | 76.31               | 94.58       | 96.24  | 80.97    | 83.65    | 71.07               | 92.92       | 94.03  | 88.82   | 83.75  |

**MRI**
| Dice score (%)           | aorta  | duodenum | esophagus | gallbladder | left adrenal gland | left kidney | liver  | pancreas | IVC | right adrenal gland | right kidney | spleen | stomach | Average   |
|-----------------|--------|----------|-----------|-------------|---------------------|-------------|--------|----------|----------|---------------------|-------------|--------|---------|--------|
| MedImageParse 3D  | 95.73 | 76.03   | 81.38    | 66.58      | 63.35              | 96.92      | 97.65 | 88.70   | 87.26   | 68.14              | 96.69      | 96.88 | 88.93  | 84.94 |
| nnU-Net        | 95.64 | 66.78   | 73.62    | 66.32      | 57.15              | 95.82      | 97.25 | 79.29   | 90.66   | 53.29              | 85.48      | 96.66 | 88.80  | 80.52 |



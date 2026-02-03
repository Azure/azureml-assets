<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `note.md` is highly recommended, but not required. It captures information about how your model is created. We highly recommend including this section to provide transparency for the customers. -->

## Intended Use

### Primary Use Cases

Feature extraction from 224 × 224 histopathology image patches. Hibou-B is intended to be used as a frozen encoder to generate patch-level embeddings for downstream tasks. These embeddings can be used for classification, clustering, retrieval, or aggregated to support slide-level analyses.

### Out-of-Scope Use Cases

Hibou-B is not designed or validated for clinical decision-making without human oversight. It is not intended for use in real-time, life-critical systems or in workflows that require regulatory-approved diagnostic tools. Developers should independently validate performance in their specific use case and comply with relevant medical regulations.

## Responsible AI Considerations

Like other vision models, Hibou-B may exhibit performance disparities when applied to image domains significantly different from its training distribution. Variations in staining protocols, scanner types, or tissue preparation may impact accuracy. Developers should evaluate the model on representative data and consider additional domain adaptation or fine-tuning where appropriate. The model has not been explicitly audited for fairness or bias in diagnostic outcomes across demographic groups or tissue types.

## Training Data

Hibou-B was trained on a large corpus of 1.3 million whole slide images (WSIs) sourced from human and veterinary pathology cases. The training data includes:

1. 936,441 H&E-stained slides  
2. 202,464 non-H&E slides  
3. 2,676 cytology slides  

These were preprocessed into 512 million clean patches using Otsu thresholding and were augmented with stain normalization, color jittering, and geometric transforms. The dataset covers over 300,000 unique cases from a wide range of diagnostic categories and anatomical sites.
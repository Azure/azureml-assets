Most medical imaging AI today is narrowly built to detect a small set of individual findings on a single modality like chest X-rays. This training approach is data- and computationally inefficient, requiring ~6-12 months per finding1, and often fails to generalize in real world environments. By further training existing multimodal foundation models on medical images and associated text data, Microsoft and Nuance created a multimodal foundation model that shows evidence of generalizing across various medical imaging modalities, anatomies, locations, severities, and types of medical data. The training methods learn to map the medical text and images into a unified numerical vector representation space, which makes it easy for computers to understand the relationships between those modalities.

Embeddings are an important building block in AI research and development for retrieval, search, comparison, classification, and tagging tasks, and developers and researchers can now use MedImageInsight embeddings in the medical domain. MedImageInsight embeddings is open source allowing developers to customize and adapt to their specific use cases.

This repository contains the MedImageInsight model, which is packaged in MLflow format and deployed using Azure ML service. The estimated time to package and deploy the model is approximately 1 hour.

This model is intended and provided as-is for research and model development exploration. MedImageInsight is not designed or intended to be deployed in clinical settings as-is nor is it for use in the diagnosis or treatment of any health or medical condition, and the model’s performance for such purposes has not been established. 

You bear sole responsibility and liability for any use of MedImageInsight, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals. 

Please see https://aka.ms/medimageinsightpaper for more details.

For documentation and example Jupyter Notebooks, visit: https://aka.ms/MedImageInsightDocs.

[^1]: [2022.12.07.22283216v3.full.pdf (medrxiv.org)](https://www.medrxiv.org/content/10.1101/2022.12.07.22283216v3.full.pdf)

### Model Architecture

Microsoft MedImageInsight includes 360 million parameter image encoder and 252 million parameter language encoder and comes as pretrained model with fine-tuning capability. The language encoder is not run in inference for each image. It is only run once (offline) to generate classifier head. MedImageInsight is a vision language transformer and was derived from the Florence computer vision foundation model. Florence is a two-tower architecture similar to CLIP, except the DaViT architecture is used as the image encoder and the UniCL objective is used as the objective function for MedImageInsight.

Model input supports image and text input and generates vector embeddings as output. This is a static model trained on an offline dataset that is described below.

### License and where to send questions or comments about the model
The license for MedImageParse is the MIT license.
For questions or comments, please contact: hlsfrontierteam@microsoft.com

### Training information

| **Training Dataset**   | **Details**        | 
|----------------|---------------------|
| **[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)**  | Frontal chest X-rays from the training partition of the MIMIC-CXR dataset and the associated text reports. Rule-based processing was carried out to extract findings and impressions separately, or to map non-labeled report sections to the relevant sections. During training, text is randomly sampled from either the findings or the impression section. In total 203,170 images from this dataset were used.|
| **[NIH-CXR-LT](https://pubmed.ncbi.nlm.nih.gov/36318048/)**| The NIH-CXR-LT dataset contains long tail distribution categories spanning 20 disease classes for frontal chest X-rays. 68,058 images from the training dataset were leveraged.      | 
| **[IRMA 2009](https://publications.rwth-aachen.de/record/113524/files/Lehmann_IRMACode_2003.pdf)**  | A dataset containing X-rays covering a spectrum of body regions, views, and patient positions. Category information is specified in a coding system, with a PDF mapping the coding system to text for each of the code sub-parts. We converted the coding scheme to the text counterparts by extracting this mapping from the PDF, and leveraged the image and code-text pairs for training.      | 
| **[RSNA BoneAge](https://pubs.rsna.org/doi/abs/10.1148/radiol.2018180736?journalCode=radiology)**  | Pediatric bone-age hand X-rays annotated with the development age of the images. The images are supplied in 8-bit format with inconsistent window leveling. Preprocessing was applied including histogram equalization followed by window leveling to control and standardize the appearance of the images for subsequent training and inference. The development age and gender of the image was converted to text using a standardized template. 12,611 images from the training partition are leveraged. |     
| **[UPENN](https://www.nature.com/articles/s41597-022-01560-7)**  | A dataset of MRI images of glioblastomas. Images were paired with the text of their DICOM image series descriptions. In total 4,645 images with associated texts were organized for training. |      
| **[TCGA](https://www.cancerimagingarchive.net/collection/tcga-sarc/)**  | multi-modal dataset of imaging for sarcoma diagnostics. CT and MRI images were extracted and associated with the text of their series description, constituting 5,643 image and text pairs. |    
| **[SD198](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_13)**  | A dataset of clinical photographs of 198 skin lesions crawled from the web. Train and test splits were not made available but based on random 50% sampling, which we followed for consistency, yielding 3,253 images for training. | 
| **[ISIC2019](https://arxiv.org/abs/1902.03368)**  | A collection of dermascopic images of skin lesions, associated with 8 diagnostic states spanning metastatic and non-metastatic disease. 20,268 images from the training partition were leveraged. | 
| **[PatchCamelyon](https://jamanetwork.com/journals/jama/fullarticle/2665774)**  | Histopathological images of breast tissue depicting the presence or absence of cancer. 262,144 images and associated text labels were used in training.  | 
| **[RSNA Mammography](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data)**  | Images from RSNA hosted and managed challenge on breast cancer detection from mammography. The dataset comprises several styles of mammo- grams with varying window levels and contrasts. No attempt was made to standardize or normalize the images. In total, 43,764 mammograms were leveraged for training. |
| **[LIDIC-IDRI](https://ieee-dataport.org/documents/lung-image-database-consortium-image-collection-lidc-idri)**  | A dataset of chest CTs depicting lung nodules at various stages of development. Dataset was broken into tiles of 5x5 across images, with tiles labeled for the maturity of lung nodule present in the tile. 80,201 tiles were sampled for training. |  
| **[PAD-UFES-20](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7479321/)**  | A collection of clinical photographs of skin lesions taken from mo- bile devices, where the images have been cropped over the lesion of interest. 6 diseases are represented. According to precedent 2,065 images (90%) were leveraged for training, and 233 (10%) for testing.  |  
| **[ODIR-5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)**  | Fundus images, where pairs of eyes were annotated across 6 categories. If one eye is not normal, the pair is labeled with the disease of the abnormal eye. Laterality specific textual descriptions were also available. Upon further processing, we discovered about 79 unique textual descriptions were assigned across 6,495 unique eyes, and opted to use these descriptions as labels instead of the reduced 6 labels. 5228 images were used for training, and 1267 images were used for evaluation, which constituted a random 20% sampling of the top 30 categories (with 10 or more instances in the dataset). |  
| **Propiertary datasets**  | Multiple other proprietary datasets, composed of procured data, data supplied by collaborative partners, and data crawled from the web were additionally leveraged for training. Caution was taken to ensure there was no leakage of test data samples in the crawled data used for training. |  


| **Carbon Footprint**                          | **Details**                                                                                                                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Carbon Footprint**                  | Pretraining utilized a cumulative 7680 GPU hours of computation on hardware of type V100 (TDP of 250W-400W). Estimated total emissions were 0.89184 tCO2eq. We trained on Azure Machine Learning. We used 64 V100 GPUs. Compute region was West US 2. |


### Evaluation Results
In this section, we report the results for the models on standard academic benchmarks. For all the evaluations, we use our internal evaluations library. For these models, we always pick the best score between our evaluation framework and any publicly reported results. Full details at https://aka.ms/medimageinsightpaper  
| **Modality**   | **Use Case**        | **Benchmark (# Labels) **                                                                                   | **Maturity relative to Human Expert** | **MSFT IP or Partner Models**   | **Google Models**                      |
|----------------|---------------------|-----------------------------------------------------------------------------------------------|--------------------------------------|---------------------------------|---------------------------------------|
| **Radiology**  | Classification      | X-Ray: RSNA Bone age                                                                           | 🟢                                 | 6.19 Ab L1*                      | No test results                       |
|                | Classification      | X-Ray: MGB Bone age                                                    | 🟢                                 | 6.57 Ab. L1                       | No test results                       |
|                | Classification      | X-Ray: IRMA2005 body-region/view categories (137)                                                    | 🟢                                 | 0.99 mAUC*                       | No test results                       |
|                | Classification      | Chest X-Ray: LT-CXR (20)                                                                                       | 🟡                                 | 0.85 mAUC                  | No test results                              |
|                | Classification      | Chest X-Ray: MGB CXR (80)                                                                                       | 🟡                                 | 0.94 mAUC                  | No test results                              |
|                | Classification      | ChestXray14: Consolidation (finetuning)                                                                | 🟡                                 | 0.74 mAUC*                |        0.74 mAUC (ELiXR)*                               |
|                | Classification      | ChestXray14: Edema (finetuning)                                                                | 🟡                                 | 0.86 mAUC*                       | 0.85 mAUC* (ELiXR)                    |
|                | Classification      | ChestXray14: Effusion (finetuning)                                                             | 🟡                                 | 0.83 mAUC*                       | 0.83 mAUC* (ELiXR)                    |
|                | Classification      | MR/CT: Exam categories (21)                                                                         | 🟡                                 | 0.95 mAUC*                       | No test results                       |
|                | Classification      | Chest CT: LIDC-IDRI Lung Nodules (4)                                                               | 🟡                                 | 0.81 mAUC*                       | No model                              |
|                | Classification      | Mammography: RSNA Mammography (4)                                                                 | 🟡                                 | 0.81 mAUC*                       | No model                              |
|                | Classification      | US: USI (3)                                                                 | 🟡                                 | 0.99 mAUC                       | No model                              |
|                | Classification      | US: HMC-QU View (2)                                                                 | 🟡                                 | 0.99 mAUC                       | No model                              |
|                | Classification      | US: Bing Echo View (7)                                                                 | 🟡                                 | 0.94 mAUC                       | No model                              |
| **Dermatology**| Classification      | ISIC2019 (8)                                                                                       | 🟡                                  | 0.97 mAUC*                       | No test results                       |
|                | Classification      | SD-198 (198)                                                                                        | 🟡                                  | 0.99 mAUC*                       | No test results                    |
|                | Classification      | PADUFES20 (6)                                                                                     | 🟡                                 | 0.95 mAUC                       | 0.97* (Med-PaLM-M 84B)                |
| **Pathology**  | Classification      | PCAM (2)                                                                                           | 🟡                                  | 0.96 mAUC*                 | No test results                       |
| **Ophthalmology**  | Classification      | OCT2017 (4)                                                                                           | 🟡                                  | 1.00 mAUC*                | No test results                       |
|                | Classification      | OCT2018 (4)                                                                                          | 🟡                                 | 1.00 mAUC*                  | No test results                              |
|                | Classification      | Fundus ODIR5K (79)                                                                                         | 🟡                                 | 0.95 mAUC                  | No test results                              |

*SOTA for this task

### Fairness evaluation

The table below highlights the performance (AUC) of Bone Age prediction and ChextX-ray text search tasks for female and male respectively.

| Tasks                                  | AUC    |
|----------------------------------------|--------|
| Bone Age (Female)                      | 6.9343 |
| Bone Age (Male)                        | 6.5446 |
| ChestX-ray text search (Female)        | 0.8651 |
| ChestX-ray text search (Male)          | 0.8603 |


The table below highlight characterisitcs of patients whose OCT images were included in the analysis. 

| Diagnosis                      | Diabetic Macular Edema (DME) | Choroidal Neovascularization (CNV) | Drusen | Normal |
|--------------------------------|------------------------------|------------------------------------|--------|--------|
| **Number of Patients**         | 709                          | 791                                | 713    | 3548   |
| **Mean Age (years)**           | 57 (Range: 20-90)            | 83 (Range: 58-97)                  | 82 (Range: 40-95) | 60 (Range: 21-86) |
| **Gender**                     |                              |                                    |        |        |
| Male                           | 38.3%                        | 54.2%                              | 44.4%  | 59.2%  |
| Female                         | 61.7%                        | 45.8%                              | 55.6%  | 40.8%  |
| **Ethnicity**                  |                              |                                    |        |        |
| Caucasian                      | 42.6%                        | 83.3%                              | 85.2%  | 59.9%  |
| Asian                          | 23.4%                        | 6.3%                               | 8.6%   | 21.1%  |
| Hispanic                       | 23.4%                        | 8.3%                               | 4.9%   | 10.2%  |
| African American               | 4.3%                         | 2.1%                               | 1.2%   | 1.4%   |
| Mixed or Other                 | 10.6%                        | 0%                                 | 0%     | 7.5%   |


We plan on doing more comprehensive fairness evaluations before public release. 

### Ethical Considerations and Limitations 

Microsoft believes Responsible AI is a shared responsibility and we have identified six principles and practices help organizations address risks, innovate, and create value: fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant use case and addresses unforeseen product misuse.   

While testing the model with images and/or text, ensure the the data is PHI free and that there are no patient information or information that can be tracked to a patient identity.

The model is not designed for the following use cases:
* **Use by clinicians to inform clinical decision-making, as a diagnostic tool, or as a medical device** - MedImageInsight is not designed or intended to be deployed as-is in clinical settings nor is it for use in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions (including to support clinical decision-making), or as a substitute of professional medical advice, diagnosis, treatment, or clinical judgment of a healthcare professional.     


* **Scenarios without consent for data** - Any scenario that uses health data for a purpose for which consent was not obtained.   

* **Use outside of health scenarios** - Any scenario that uses non-medical related image and/or serving purposes outside of the healthcare domain.   

Please see Microsoft's Responsible AI Principles and approach available at [https://www.microsoft.com/en-us/ai/principles-and-approach/](https://www.microsoft.com/en-us/ai/principles-and-approach/)


### Sample inputs and outputs (for real time inference)

**Input:**
```bash
data =  {
  "input_data": {
    "columns": [
      "image",
      "text"
    ],
    "index":[0],
    "data": [
      [base64.encodebytes(read_image(sample_image_1)).decode("utf-8"), "x-ray chest anteroposterior Cardiomegaly"]
    ]
  },
  "params":{
      "get_scaling_factor": True
  }
}
```

**Output:**
```bash
[
  {
    "image_features": [
      [-0.040428221225738525, 0.015632804483175278, -0.034625787287950516, -0.013094332069158554, ... , 0.023215821012854576, -0.010303247720003128, -0.003998206462711096, -0.00022746287868358195]
    ]
  },
  {
    "text_features": [
      [-0.04121647855639458, 0.014923677921295166, -0.033598374396562576, -0.012765488520264626, ... ,  0.02294582130014801, -0.009835227608680725, -0.004232016112744808, -0.00021812367581298325]
    ]
  },
  {
    "scaling_factor": 4.513362407684326
  }
]
```

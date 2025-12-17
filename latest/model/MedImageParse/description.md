Biomedical image analysis is fundamental for biomedical discovery in cell biology, pathology, radiology, and many other biomedical domains. MedImageParse is a biomedical foundation model for imaging parsing that can jointly conduct segmentation, detection, and recognition across 9 imaging modalities. Through joint learning, we can improve accuracy for individual tasks and enable novel applications such as segmenting all relevant objects in an image through a text prompt, rather than requiring users to laboriously specify the bounding box for each object. 

MedImageParse is broadly applicable, performing image segmentation across 9 imaging modalities. 

MedImageParse is also able to identify invalid user inputs describing objects that do not exist in the image. MedImageParse can perform object detection, which aims to locate a specific object of interest, including on objects with irregular shapes. 

On object recognition, which aims to identify all objects in a given image along with their semantic types, MedImageParse can simultaneously segment and label all biomedical objects in an image. 

In summary, MedImageParse shows potential to be a building block for an all-in-one tool for biomedical image analysis by jointly solving segmentation, detection, and recognition. 

It is broadly applicable to all major biomedical image modalities, which may pave a future path for efficient and accurate image-based biomedical discovery when built upon and integrated into an application.

This repository contains the MedImageParse model, which is packaged in MLflow format and deployed using Azure ML service. The estimated time to package and begin to build upon the model is approximately 1 hour.

This model is intended and provided as-is for research and model development exploration. MedImageParse is not designed or intended to be deployed in clinical settings as-is nor is it intended for use in the diagnosis or treatment of any health or medical condition, and the model’s performance for such purposes has not been established. You bear sole responsibility and liability for any use of MedImageParse, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals.

For documentation and example Jupyter Notebooks, visit: https://aka.ms/MedImageParseDocs.

For example code, usage demonstrations, and fine-tuning capabilities, visit the [Healthcare AI Examples repository](https://aka.ms/HealthcareAIExamples).

### Model Architecture
MedImageParse is built upon a transformer-based architecture, optimized for processing large biomedical corpora. Leveraging multi-head attention mechanisms, it excels at identifying and understanding biomedical terminology, as well as extracting contextually relevant information from dense scientific texts. The model is pre-trained on vast biomedical datasets, allowing it to generalize across various biomedical domains with high accuracy.

### License and where to send questions or comments about the model
The license for MedImageParse is the MIT license. Please cite our paper if you use the model for your research https://microsoft.github.io/BiomedParse/assets/BiomedParse_arxiv.pdf and https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Boltzmann_Attention_Sampling_for_Image_Analysis_with_Small_Objects_CVPR_2025_paper.pdf .
For questions or comments, please contact: hlsfrontierteam@microsoft.com

### Training information

MedImageParse was trained on a large dataset comprising over six million triples of image, segmentation mask, and textual description.

MedImageParse used 40 NVIDIA A100-SXM4-80GB GPUs for a duration of 48 hours. 

### Evaluation Results
Please see the paper for detailed information about methods and results. https://microsoft.github.io/BiomedParse/assets/BiomedParse_arxiv.pdf as well as https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Boltzmann_Attention_Sampling_for_Image_Analysis_with_Small_Objects_CVPR_2025_paper.pdf.

Bar plot comparing the Dice score between our method and competing methods on 102,855 test instances (image-mask-label
triples) across 9 modalities. MedSAM and SAM require bounding box as input. 

<img src="https://automlcesdkdataresources.blob.core.windows.net/model-cards/model_card_images/MedImageParse/medimageparseresults.png" alt="MedImageParse comparison results on segmentation">

### Fairness evaluation
We conducted fairness evaluation for different sex and age groups. Two-sided independent t-test 
shows non-significant differences between female and male and between different age groups, with p-value > 5% for all imaging modalities and segmentation targets evaluated.

### Ethical Considerations and Limitations 

Microsoft believes Responsible AI is a shared responsibility and we have identified six principles and practices to help organizations address risks, innovate, and create value: fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant use case and addresses unforeseen product misuse.   

While testing the model with images and/or text, ensure that the data is PHI free and that there are no patient information or information that can be tracked to a patient identity.

The model is not designed for the following use cases:
* **Use by clinicians to inform clinical decision-making, as a diagnostic tool or as a medical device** - Although MedImageParse is highly accurate in parsing biomedical data, it is not desgined or intended to be deployed in clinical settings as-is not is it for use in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions (including to support clinical decision-making), or as a substitute of professional medical advice, diagnosis, treatment, or clinical judgment of a healthcare professional.  

* **Scenarios without consent for data** - Any scenario that uses health data for a purpose for which consent was not obtained.   

* **Use outside of health scenarios** - Any scenario that uses non-medical related image and/or serving purposes outside of the healthcare domain.   

Please see Microsoft's Responsible AI Principles and approach available at [https://www.microsoft.com/en-us/ai/principles-and-approach/](https://www.microsoft.com/en-us/ai/principles-and-approach/)



### Sample inputs and outputs (for real time inference)

Input:
```bash
data = {
  "input_data": {
    "columns": [
      "image",
      "text"
    ],
    "index":[0, 1],
    "data": [
      [base64.encodebytes(read_image('./examples/Part_3_226_pathology_breast.png')).decode("utf-8"), "neoplastic cells in breast pathology & inflammatory cells."],
      [base64.encodebytes(read_image('./examples/TCGA_HT_7856_19950831_8_MRI-FLAIR_brain.png')).decode("utf-8"), "brain tumor"]
    ],
  },
  "params": {}
}
```



## Data and Resource Specification for Deployment
* **Supported Data Input Format** 
1. The model expect 2D 8-bit RGB or grayscale images by default, with pixel values ranging from 0 to 255 and resolution 1024*1024. 
2. We provided preprocessing notebooks 4, 5, 6 to illustrate how to convert raw formats including DICOM, NIFTI, PNG, and JPG to desired format, with preprocessing steps such as CT windowing.
3. The model outputs pixel probabilities in the same shape as the input image. We convert the floating point probabilities to 8-bit grayscale outputs. The probability threshold for segmentation mask is 0.5, which corresponds to 127.5 in 8-bit grayscale output.
4. The model takes in text prompts for segmentation and doesn't have a fixed number of targets to handle. However, to ensure quality performance, we recommend the following tasks based on evaluation results.
  - CT: abdomen: adrenal gland, aorta, bladder, duodenum, esophagus, gallbladder, kidney, kidney cyst, 
            kidney tumor, left adrenal gland, left kidney, liver, pancreas, postcava, 
            right adrenal gland, right kidney, spleen, stomach, tumor
        colon: tumor
        liver: liver, tumor
        lung: COVID-19 infection, nodule
        pelvis: uterus 
  - MRI-FLAIR: brain: edema, lower-grade glioma, tumor, tumor core, whole tumor
  - MRI-T1-Gd: brain: enhancing tumor, tumor core
  - MRI-T2: prostate: prostate peripheral zone, prostate transitional zone, 
  - MRI: abdomen: aorta, esophagus, gallbladder, kidney, left kidney, liver, pancreas, postcava, 
                right kidney, spleen, stomach 
        brain: anterior hippocampus, posterior hippocampus
        heart: left heart atrium, left heart ventricle, myocardium, right heart ventricle
        prostate: prostate 
  - OCT: retinal: edema
  - X-Ray: chest: COVID-19 infection, left lung, lung, lung opacity, right lung, viral pneumonia 
  - dermoscopy: skin: lesion, melanoma
  - endoscope: colon: neoplastic polyp, non-neoplastic polyp, polyp 
  - fundus: retinal: optic cup, optic disc, 
  - pathology: bladder: neoplastic cells
            breast: epithelial cells, neoplastic cells
            cervix: neoplastic cells
            colon: glandular structure, neoplastic cells
            esophagus: neoplastic cells
            kidney: neoplastic cells
            liver: epithelial cells, neoplastic cells
            ovarian: epithelial cells, 'neoplastic cells
            prostate: neoplastic cells
            skin: neoplastic cells
            stomach: neoplastic cells
            testis: epithelial cells
            thyroid: epithelial cells, neoplastic cells 
            uterus: neoplastic cells
ultrasound: breast: benign tumor, malignant tumor, tumor
            heart: left heart atrium, left heart ventricle
            transperineal: fetal head, public symphysis


## Version History

| Version | Date | Description |
|---------|------|-------------|
| 16 | 2025-12-10 | - Initial version tracking <br>- Remove V100 support|
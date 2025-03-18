<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `note.md` is highly recommended, but not required. It captures information about how your model is created. We highly recommend including this section to provide transparency for the customers. -->

## Intended Use

### Primary Use Cases

* **Supported Data Input Format** 
1. The model expect 3D NIfTI images by default. 
2. The model outputs pixel probabilities in the same shape as the input image. The probability threshold for segmentation mask is 0.5.
3. The model takes in text prompts for segmentation and doesn't have a fixed number of targets to handle. However, to ensure quality performance, we recommend the following tasks based on evaluation results. Wil will extend the model capability with more object types including tumors and nodules.
  - CT: abdomen: adrenal gland, aorta, bladder, duodenum, esophagus, gallbladder, kidney,
            left adrenal gland, left kidney, liver, pancreas, postcava, 
            right adrenal gland, right kidney, spleen, stomach
  - MRI: abdomen: aorta, esophagus, gallbladder, kidney, left kidney, liver, pancreas, postcava, 
                right kidney, spleen, stomach 
        

### Out-of-Scope Use Cases
This model is intended and provided as-is for research and model development exploration. MedImageParse 3D is not designed or intended to be deployed in clinical settings as-is nor is it intended for use in the diagnosis or treatment of any health or medical condition, and the model’s performance for such purposes has not been established. 
You bear sole responsibility and liability for any use of MedImageParse 3D, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals. When evaluating the model for your use case, carefully consider the impacts of overreliance, including overreliance within the context of radiology specifically and more generally for generative AI [Appropriate reliance on Generative AI: Research synthesis - Microsoft Research]([https://www.microsoft.com/en-us/research/publication/appropriate-reliance-on-generative-ai-research-synthesis/])

## Responsible AI Considerations
Microsoft believes Responsible AI is a shared responsibility and we have identified six principles and practices to help organizations address risks, innovate, and create value: fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant use case and addresses unforeseen product misuse.   

While testing the model with images and/or text, ensure that the data is PHI free and that there are no patient information or information that can be tracked to a patient identity.

The model is not designed for the following use cases:
* **Use by clinicians to inform clinical decision-making, as a diagnostic tool or as a medical device** - Although MedImageParse 3D is highly accurate in parsing biomedical data, it is not designed or intended to be deployed in clinical settings as-is not is it for use in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions (including to support clinical decision-making), or as a substitute of professional medical advice, diagnosis, treatment, or clinical judgment of a healthcare professional.  

* **Scenarios without consent for data** - Any scenario that uses health data for a purpose for which consent was not obtained.   

* **Use outside of health scenarios** - Any scenario that uses non-medical related image and/or serving purposes outside of the healthcare domain.   

Please see Microsoft's Responsible AI Principles and approach available at [https://www.microsoft.com/en-us/ai/principles-and-approach/](https://www.microsoft.com/en-us/ai/principles-and-approach/)


## Training Data

The training data include AMOS22-CT, AMOS22-MRI.


### License and where to send questions or comments about the model
The license for MedImageParse 3D is the MIT license. Please cite our [`Paper`](https://aka.ms/biomedparse-paper) if you use the model for your research.
For questions or comments, please contact: hlsfrontierteam@microsoft.com

### Citation
Zhao, T., Gu, Y., Yang, J. et al. A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02499-w

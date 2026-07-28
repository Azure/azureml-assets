<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `note.md` is highly recommended, but not required. It captures information about how your model is created. We highly recommend including this section to provide transparency for the customers. -->

## Intended Use

### Primary Use Cases

* **Supported Data Input Format**

1. The model expect 3D NIfTI images by default.
2. The model outputs pixel probabilities in the same shape as the input image. The probability threshold for segmentation mask is 0.5.
3. The model takes in text prompts for segmentation and doesn't have a fixed number of targets to handle. However, to ensure quality performance, we recommend the following tasks based on evaluation results. Wil will extend the model capability with more object types including tumors and nodules.

* **CT:** oncology/pathology (adrenocortical carcinoma, kidney lesions/cysts L/R, liver tumors, lung lesions, pancreas tumors, head–neck cancer, colon cancer primaries, COVID-19, whole-body lesion, lymph nodes); thoracic (lungs L/R, lobes LUL/LLL/RUL/RML/RLL, trachea, airway tree); abdomen/pelvis (spleen, liver, gallbladder, stomach, pancreas, duodenum, small bowel, colon, esophagus); GU/endocrine (kidneys L/R, adrenal glands L/R, bladder, prostate, uterus); vascular (aorta/tree, SVC, IVC, pulmonary vein, brachiocephalic trunk, subclavian/carotid arteries L/R, brachiocephalic veins L/R, left atrial appendage, portal/splenic vein, iliac arteries/veins L/R); cardiac (heart); head/neck (carotids L/R, submandibular/parotid/lacrimal glands L/R, thyroid, larynx glottic/supraglottic, lips, buccal mucosa, oral cavity, cervical esophagus, cricopharyngeal inlet, arytenoids, eyeball segments ant/post L/R, optic chiasm, optic nerves L/R, cochleae L/R, pituitary, brainstem, spinal cord); neuro/cranial (brain, skull, Circle of Willis CTA); spine/MSK (sacrum, vertebrae C1–S1, humeri/scapulae/clavicles/femora/hips L/R, gluteus maximus/medius/minimus L/R, autochthon L/R, iliopsoas L/R).
* **MRI:** abdomen/pelvis (spleen, liver, gallbladder, stomach, pancreas, duodenum, small bowel, colon whole, esophagus, bladder, prostate, uterus); colon segments (cecum, appendix, ascending, transverse, descending, sigmoid, rectum); GU (prostate transition zone, prostate lesion); cardiac CMR (LV, RV, myocardium, LA, RA); thoracic (lungs L/R); vascular (aorta, pulmonary artery, SVC, IVC, portal/splenic vein, iliac arteries/veins L/R, carotid arteries L/R, jugular veins L/R); neuro tumors/ischemia (brain, brain tumor, stroke lesion, GTVp/GTVn tumor, vestibular schwannoma intra/extra-meatal, cochleae L/R); glioma components (non-enhancing tumor core, non-enhancing FLAIR hyperintensity, enhancing tissue, resection cavity); white matter disease (WM hyperintensities FLAIR/T1); neurovascular (Circle of Willis MRA); spine/MSK (sacrum, vertebrae regional, discs, spinal canal/cord, humeri/femora/hips L/R, gluteus maximus/medius/minimus L/R, autochthon L/R, iliopsoas L/R).
* **Ultrasound:** cardiac (LV, myocardium, LA), neck (thyroid, carotid artery, jugular vein), neuro (brain tumor), calf MSK (soleus, gastrocnemius medialis/lateralis).
* **PET / PET-CT:** whole-body lesion.
* **Electron Microscopy:** endolysosomes, mitochondria, nuclei, neuronal ultrastructure, synaptic clefts, axon.
* **Light-Sheet Microscopy:** brain neural activity, Alzheimer’s plaque, nuclei, vessel.

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

Data description:

The dataset covers five commonly used 3D biomedical image modalities: CT, MR, PET, Ultrasound, and Microscopy. All the images are from public datasets with a License for redistribution. All images were processed to npz format with an intensity range of [0, 255]. Specifically, for CT images, the Hounsfield units were normalized using typical window width and level values: soft tissues (W:400, L:40), lung (W:1500, L:-160), brain (W:80, L:40), and bone (W:1800, L:400). Subsequently, the intensity values were rescaled to the range of [0, 255]. For other images, the intensity values were clipped in the range between the 0.5th and 99.5th percentiles before rescaling them to the range of [0, 255]. If the original intensity range is already in [0, 255], no preprocessing was applied.

### License and where to send questions or comments about the model

The license for MedImageParse 3D is the MIT license. Please cite our [`Paper`](https://aka.ms/biomedparse-paper) if you use the model for your research.
For questions or comments, please contact: hlsfrontierteam@microsoft.com

### Citation

Zhao, T., Gu, Y., Yang, J. et al. A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02499-w

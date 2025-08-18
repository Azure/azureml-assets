Most medical imaging AI today is narrowly built to detect a small set of individual findings on a single modality like chest X-rays. This training approach is data- and computationally inefficient, requiring ~6-12 months per finding[1], and often fails to generalize in real world environments. By further training existing multimodal foundation models on medical images and associated text data, Microsoft and Nuance created a multimodal foundation model that shows evidence of generalizing across various medical imaging modalities, anatomies, locations, severities, and types of medical data. The training methods learn to map the medical text and images into a unified numerical vector representation space, which makes it easy for computers to understand the relationships between those modalities.

Embeddings are an important building block in AI research and development for retrieval, search, comparison, classification, and tagging tasks, and developers and researchers can now use MedImageInsight embeddings in the medical domain. MedImageInsight embeddings is open source allowing developers to customize and adapt to their specific use cases.

This model is intended and provided as-is for research and model development exploration. MedImageInsight is not designed or intended to be deployed in clinical settings as-is nor is it for use in the diagnosis or treatment of any health or medical condition, and the model's performance for such purposes has not been established. 

This model is the ONNX format of the original model **MedImageInsight** (`azureml://registries/azureml/models/MedImageInsight`) which is available at Azure AI Foundry Model catalog as well. **MedImageInsight-onnx** model has been optimized in terms of the inference speed while keeping the same accuracy as the original model.

You bear sole responsibility and liability for any use of MedImageInsight-onnx, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals. 

Please see https://aka.ms/medimageinsightpaper for more details.

[1]: [2022.12.07.22283216v3.full.pdf (medrxiv.org)](https://www.medrxiv.org/content/10.1101/2022.12.07.22283216v3.full.pdf)

### Model Speed-up
The model is optimized for inference speed using ONNX Runtime.

#### Datasets
- **LTCXR**: LongtailCXR Test - 20k test samples,
- **RSNA-M**: RSNA Mammography STD - 10k test samples
- **RSNA-B-Male**: RSNA Bone Age Male - 773 test samples, 256 total classes, 65 classes used in test
- **Gastrovision**: 3.2k test samples, 27 classes
- **MGBCXR**: 684 test samples, 80 classes used in test

#### Models
- **Baseline (fp16)**: the original pytorch model.
- **OnnxRuntime (fp16)**: ONNX conversion of original model, with optimizations specific for OnnxRuntime.

#### Metrics
- **Latency**: The time taken for the model to process a single input sample in milliseconds.
- **Speed-up**: The percentage reduction in latency compared to the baseline model.
- **Accuracy**: The percentage of correct predictions made by the model.
- **BACC**: Balanced Accuracy, which accounts for class imbalance in the dataset.
- **mAUC**: Mean Area Under the Curve, a measure of the model's ability to distinguish between classes.

#### Results
The table below shows the inference latency, speed-up, and accuracy of the model on various datasets. The speed-up is calculated as the percentage reduction in latency compared to the baseline model.

|**Dataset**        |**Model**        |**Latency** (ms) |**Speed-up** |**Accuracy** |**BACC**    |**mAUC**    |
|-------------------|-----------------|-----------------|-------------|-------------|------------|------------|
| **LTCXR**         | **Baseline**    | 137             |  -          | 11.31%      | 39.38%     | 87.73%     |
|                   | OnnxRuntime     | 45              | **61.84%**  | 11.13%      | 39.34%     | 87.72%     |
| **RSNA-M**        | **Baseline**    | 85              |  -          | 61.84%      | 66.5%      | 88.83%     |
|                   | OnnxRuntime     | 34              | **60.00%**  | 62.26%      | 66.69%     | 88.85%     |
| **RSNA-B-Male**   | **Baseline**    | 270             |  -          | 26.78%      | 15.26%     | 93.47%     |
|                   | OnnxRuntime     | 73              | **72.96%**  | 26.65%      | 15.11%     | 93.47%     |
| **Gastrovision**  | **Baseline**    | 160             |  -          | 20.92%      | 11.17%     | 77.61%     |
|                   | OnnxRuntime     | 50              | **68.75%**  | 21.04%      | 11.26%     | 76.77%     |
| **MGBCXR**        | **Baseline**    | 322             |  -          | 46.93%      | 34.61%     | 92.55%     |
|                   | OnnxRuntime     | 84              | **73.91%**  | 46.93%      | 34.61%     | 92.75%     |

### Ethical Considerations and Limitations 

Microsoft believes Responsible AI is a shared responsibility, and we have identified six principles and practices that help organizations address risks, innovate, and create value: fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant use case and addresses unforeseen product misuse.

While testing the model with images and/or text, ensure the data is PHI free and that there are no patient information or information that can be tracked to a patient identity.

The model is not designed for the following use cases:
* **Use by clinicians to inform clinical decision-making, as a diagnostic tool, or as a medical device** - MedImageInsight is not designed or intended to be deployed as-is in clinical settings nor is it for use in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions (including to support clinical decision-making), or as a substitute of professional medical advice, diagnosis, treatment, or clinical judgment of a healthcare professional.


* **Scenarios without consent for data** - Any scenario that uses health data for a purpose for which consent was not obtained.

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
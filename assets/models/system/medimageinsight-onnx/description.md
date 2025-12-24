Most medical imaging AI today is narrowly built to detect a small set of individual findings on a single modality like chest X-rays. This training approach is data- and computationally inefficient, requiring ~6-12 months per finding[1], and often fails to generalize in real world environments. By further training existing multimodal foundation models on medical images and associated text data, Microsoft and Nuance created a multimodal foundation model that shows evidence of generalizing across various medical imaging modalities, anatomies, locations, severities, and types of medical data. The training methods learn to map the medical text and images into a unified numerical vector representation space, which makes it easy for computers to understand the relationships between those modalities.

Embeddings are an important building block in AI research and development for retrieval, search, comparison, classification, and tagging tasks, and developers and researchers can now use MedImageInsight embeddings in the medical domain. MedImageInsight embeddings is open source allowing developers to customize and adapt to their specific use cases.

This model is intended and provided as-is for research and model development exploration. MedImageInsight is not designed or intended to be deployed in clinical settings as-is nor is it for use in the diagnosis or treatment of any health or medical condition, and the model's performance for such purposes has not been established. 

This model is the ONNX format of the original model **MedImageInsight** (`azureml://registries/azureml/models/MedImageInsight`) which is available at Azure AI Foundry Model catalog as well. **MedImageInsight-onnx** model has been optimized in terms of the inference speed while keeping the same accuracy as the original model.

You bear sole responsibility and liability for any use of MedImageInsight-onnx, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals. 

Please see https://aka.ms/medimageinsightpaper for more details.

For example code, usage demonstrations, and fine-tuning capabilities, visit the [Healthcare AI Examples repository](https://aka.ms/HealthcareAIExamples).

[1]: [2022.12.07.22283216v3.full.pdf (medrxiv.org)](https://www.medrxiv.org/content/10.1101/2022.12.07.22283216v3.full.pdf)

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


## Version History

| Version | Date | Description |
|---------|------|-------------|
| 2 | 2025-12-10 | - Initial version tracking <br>- Remove V100 support|
The Swin Transformer V2 model is a type of Vision Transformer, pre-trained on ImageNet-21k with a resolution of 192x192, is introduced in the <a href="https://arxiv.org/abs/2111.09883" target="_blank">research-paper</a> titled "Swin Transformer V2: Scaling Up Capacity and Resolution" authored by Liu et al. This model tries to resolve training instability, resolution gaps between pre-training and fine-tuning, and large labelled data issues in training and application of large vision models. This model generates hierarchical feature maps by merging image patches and computes self attention within a local window resulting in a linear computational complexity relative to input image size which is a significant improvement over vision transformers that take quadratic computational complexity.

Swin Transformer V2 introduces three improvements:
- a residual-post-norm method with cosine attention to improve training stability
- a log-spaced continuous position bias method, aiding the transfer of pre-trained models from low-resolution images to tasks with high-resolution inputs
- the application of a self-supervised pre-training method called SimMIM, designed to reduce the need for extensive labeled images

# License

Apache License, Version 2.0. For more details refer <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank">apache-2.0</a>

# Inference Samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-classification" target="_blank">image-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-classification" target="_blank">image-classification-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-image-classification" target="_blank">image-classification-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-image-classification" target="_blank">image-classification-batch-endpoint.sh</a>

# Finetuning Samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image Multi-class classification|Image Multi-class classification|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip)|<a href="https://aka.ms/azureml-ft-sdk-image-mc-classification" target="_blank">fridgeobjects-multiclass-classification.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-image-mc-classification" target="_blank">fridgeobjects-multiclass-classification.sh</a>
Image Multi-label classification|Image Multi-label classification|[multilabel fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip)|<a href="https://aka.ms/azureml-ft-sdk-image-ml-classification" target="_blank">fridgeobjects-multilabel-classification.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-image-ml-classification" target="_blank">fridgeobjects-multilabel-classification.sh</a>

# Evaluation Samples

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Image Multi-class classification|Image Multi-class classification|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip)|<a href="https://aka.ms/azureml-evaluation-sdk-image-mc-classification" target="_blank">image-multiclass-classification.ipynb</a>|
|Image Multi-label classification|Image Multi-label classification|[multilabel fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip)|<a href="https://aka.ms/azureml-evaluation-sdk-image-ml-classification" target="_blank">image-multilabel-classification.ipynb</a>|

# Sample input and output

### Sample input

```json
{
  "input_data": {
    "columns": [
      "image"
    ],
    "index": [0, 1],
    "data": ["image1", "image2"]
  }
}
```

Note: "image1" and "image2" string should be in base64 format or publicly accessible urls.


### Sample output

```json
[
    {
        "probs": [0.91, 0.09],
        "labels": ["can", "carton"]
    },
    {
        "probs": [0.1, 0.9],
        "labels": ["can", "carton"]
    }
]
```

#### Visualization of inference result for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_microsoft-swinv2-base-patch4-window12-192-22k_MC_new.png" alt="mc visualization">

Note: The labels provided by swinv2 model are class indices appended to "LABEL_"(starting from "LABEL_0" to "LABEL_21841"). For e.g. "LABEL_3500" for "Giraffe". For visualization purpose, we explictly mapped these labels to imagenet-21k class names which are shown above in the sample image.
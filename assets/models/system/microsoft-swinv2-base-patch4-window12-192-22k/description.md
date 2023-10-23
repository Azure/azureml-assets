The Swin Transformer is a type of Vision Transformer used in both image classification and dense recognition tasks. It builds hierarchical feature maps by merging image patches in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window. Previous vision Transformers produce feature maps of a single low resolution and have quadratic computation complexity to input image size due to the computation of self-attention globally. Swin Transformer v2 has three main improvements which are a residual-post-norm method, a log-spaced continuous position bias method, and a self-supervised pre-training method called SimMIM. These improvements combined with cosine attention help improve training stability and reduce the need for vast labeled images.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-classification" target="_blank">image-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-classification" target="_blank">image-classification-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-image-classification" target="_blank">image-classification-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-image-classification" target="_blank">image-classification-batch-endpoint.sh</a>

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image Multi-class classification|Image Multi-class classification|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip)|<a href="https://aka.ms/azureml-ft-sdk-image-mc-classification" target="_blank">fridgeobjects-multiclass-classification.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-image-mc-classification" target="_blank">fridgeobjects-multiclass-classification.sh</a>
Image Multi-label classification|Image Multi-label classification|[multilabel fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip)|<a href="https://aka.ms/azureml-ft-sdk-image-ml-classification" target="_blank">fridgeobjects-multilabel-classification.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-image-ml-classification" target="_blank">fridgeobjects-multilabel-classification.sh</a>

### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Image Multi-class classification|Image Multi-class classification|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip)|<a href="https://aka.ms/azureml-evaluation-sdk-image-mc-classification" target="_blank">image-multiclass-classification.ipynb</a>|
|Image Multi-label classification|Image Multi-label classification|[multilabel fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip)|<a href="https://aka.ms/azureml-evaluation-sdk-image-ml-classification" target="_blank">image-multilabel-classification.ipynb</a>|

### Sample inputs and outputs (for real-time inference)

#### Sample input

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


#### Sample output

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

#### Model inference - visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_microsoft-swinv2-base-patch4-window12-192-22k_MC.png" alt="mc visualization">

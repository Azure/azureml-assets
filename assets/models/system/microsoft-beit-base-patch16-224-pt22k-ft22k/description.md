BEiT (Bidirectional Encoder representation from Image Transformers) is a vision transformer(ViT) pre-trained with Masked Image Modeling(MIM), which is a self-supervised pre-training inspired by BERT from NLP, followed by Intermediate fine-tuning using ImageNet-22k dataset. It is then fine-tuned for Image Classification. Images have two views of representation in BEiT, image patches and visual tokens which serve as input and output during pre-training, respectively. During self-supervised pre-training stage, some percentage of image patches are masked randomly, and then the visual tokens corresponding to the masked patches are predicted.

Through pre-training, the model acquires an internal representation of images, enabling the extraction of features useful for subsequent tasks. After pre-training, a simple linear classifier layer is employed as a task layer on top of pre-trained encoder for image classification, which includes average pooling to aggregate the representations and feed the global to a softmax classifier.

> For more details, refer <a href="https://arxiv.org/abs/2106.08254" target="_blank">BEiT-paper</a>.

# Training Details

## Training Data

The BEiT model is pre-trained on ImageNet-22k, encompassing 14 million images and 21,000 classes and fine-tuned on the same dataset.

## Training Procedure

In the preprocessing step, images are resized to the same resolution 224x224. Images are scaled with augmentations like random resized cropping, horizontal flipping, color jittering. Then normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

For more details on self-supervised pre-training (ImageNet-22k) followed by supervised fine-tuning (ImageNet-1k) refer to the section 2 and 3 of the original paper.

# Evaluation results

For BEiT image classification benchmark results, Refer to the table 1 of the <a href="https://arxiv.org/abs/2106.08254" target="_blank">original-paper</a>.

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

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_microsoft-beit-base-patch16-224-pt22k-ft22k_MC.png" alt="mc visualization">

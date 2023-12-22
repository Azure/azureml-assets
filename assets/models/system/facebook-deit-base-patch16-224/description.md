DeiT (Data-efficient image Transformers) is an image transformer that do not require very large amounts of data for training. This is achieved through a novel distillation procedure using teacher-student strategy, which results in high throughput and accuracy. DeiT is pre-trained and fine-tuned on ImageNet-1k (1 million images, 1,000 classes) at resolution 224x224. The model was first released in this <a href="https://github.com/facebookresearch/deit" target="_blank">repository</a>, but the weights were converted to PyTorch from the <a href="https://github.com/huggingface/pytorch-image-models" target="_blank">timm</a> repository by Ross Wightman.

An image is treated as a sequence of patches and it is processed by a standard Transformer encoder as used in NLP. These patches are linearly embedded, and a [CLS] token is added at the beginning of the sequence for classification tasks. The model also requires absolute position embeddings before feeding the sequence Transformer encoder. So the pre-training creates an inner representation of images that can be used to extract features that are useful for downstream tasks. For instance, if a dataset of labeled images is available, a linear layer can be placed on top of the pre-trained encoder, to train a standard classifier.

> For more details on DeiT, Review the <a href="https://arxiv.org/abs/2012.12877" target="_blank">original-paper</a>.

# Training Details

## Training Data

The DeiT model is pre-trained and fine-tuned on ImageNet 2012, consisting of 1 million images and 1,000 classes on a resolution of 224x224.

## Training Procedure

In the preprocessing step, images are resized to the same resolution 224x224. Different augmentations like Rand-Augment, and random erasing are used. For more details on transformations during training/validation refer <a href="https://github.com/facebookresearch/deit/blob/main/datasets.py#L78" target="_blank">this-link</a>. At inference time, images are rescaled to the same resolution 256x256, center-cropped at 224x224 and then normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

The model was trained on a single 8-GPU node for 3 days. Training resolution is 224. For more details on hyperparameters refer to table 9 of the <a href="https://arxiv.org/abs/2012.12877" target="_blank">original-paper</a>.

For more details on pre-training (ImageNet-1k) followed by supervised fine-tuning (ImageNet-1k) refer to the section 2 to 5 of the <a href="https://arxiv.org/abs/2012.12877" target="_blank">original-paper</a>.

# Evaluation results

DeiT base model achieved top-1 accuracy of 81.8% and top-5 accuracy of 95.6% on ImageNet with 86M parameters with image size 224x224. For DeiT image classification benchmark results, refer to the table 5 of the <a href="https://arxiv.org/abs/2012.12877" target="_blank">original-paper</a>.

It's important to note that during the fine-tuning process, superior performance is attained with a higher resolution, and enhancing the model size leads to improved performance.

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

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_facebook-deit-base-patch16-224_laptop_MC.png" alt="mc visualization">

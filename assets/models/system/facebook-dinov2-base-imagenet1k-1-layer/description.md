# Vision Transformer (base-sized model) trained using DINOv2
Vision Transformer (ViT) model trained using the DINOv2 method. It was introduced in the paper <a href="https://arxiv.org/abs/2304.07193">DINOv2: Learning Robust Visual Features without Supervision by Oquab et al.</a> and first released in this <a href="https://github.com/facebookresearch/dinov2">repository</a>.

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a self-supervised fashion.\n Images are presented to the model as a sequence of fixed-size patches, which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.\n Note that this model does not include any fine-tuned heads.\n By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

> For more details on Dinov2, Review the <a href="https://arxiv.org/abs/2304.07193" target="_blank">original-paper</a> and the <a href="https://github.com/facebookresearch/dinov2" target="_blank">model's github repo</a>

The model takes an image as input and returns a class token and patch tokens, and optionally 4 register tokens.

The embedding dimension is:

<li> 384 for ViT-S.</li>
<li> 768 for ViT-B.</li>
<li> 1024 for ViT-L</li>
<li> 1536 for ViT-g</li>

The models follow a Transformer architecture, with a patch size of 14. In the case of registers, we add 4 register tokens, learned during training, to the input sequence after the patch embedding.

For a 224x224 image, this results in 1 class token + 256 patch tokens, and optionally 4 register tokens.

The models can accept larger images provided the image shapes are multiples of the patch size (14). If this condition is not verified, the model will crop to the closest smaller multiple of the patch size.

# Training Details

## Training Data

The Dinov2 model is pre-trained and fine-tuned on ImageNet 2012,  of 1 million consistingimages and 1,000 classes on a resolution of 224x224.

# License

apache-2.0

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
  "input_data": ["image1", "image2"]
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

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_facebook-dinov2-base-imagenet1k-1-layer_elephant_MC.png" alt="mc visualization">
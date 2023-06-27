The Vision Transformer (ViT) is a BERT-like transformer encoder model which is pretrained on a large collection of images in a supervised fashion, such as ImageNet-21k. The ImageNet dataset comprises 1 million images and 1000 classes at a resolution of 224x224, which the model was fine-tuned on. The model assumes that the image is presented as a sequence of fixed-size patches, which are linearly embedded, and that a [CLS] token is added to the beginning of the sequence for classification tasks. The model also requires absolute position embeddings to be added before feeding the sequence to the layers of the Transformer encoder. By pretraining the model in this way, it is possible to create an inner representation of images that can be used to extract features and classifiers useful for downstream tasks. For instance, if a dataset of labeled images is available, a linear layer can be placed on top of the pre-trained encoder, with the last hidden state of the [CLS] token serving as a representation of the entire image.

> The above summary was generated using ChatGPT. Review the [original-model-card](https://huggingface.co/google/vit-base-patch16-224) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[image-classification-online-endpoint.ipynb](https://aka.ms/azureml-infer-sdk-image-classification)|[image-classification-online-endpoint.sh](https://aka.ms/azureml-infer-cli-image-classification)
Batch | todo | todo

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image Multi-class Classifiaction|Image Multi-class Classifiaction|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip)|[fridgeobjects-multiclass-classification.ipynb](https://aka.ms/azureml-ft-sdk-image-mc-classification)|[fridgeobjects-multiclass-classification.sh](https://aka.ms/azureml-ft-cli-image-mc-classification)
Image Multi-label Classifiaction|Image Multi-label Classifiaction|[multilabel fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip)|[fridgeobjects-multilabel-classification.ipynb](https://aka.ms/azureml-ft-sdk-image-ml-classification)|[fridgeobjects-multilabel-classification.sh](https://aka.ms/azureml-ft-cli-image-ml-classification)

### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Image Multi-class Classifiaction|Image Multi-class Classifiaction|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip)|todo|
|Image Multi-label Classifiaction|Image Multi-label Classifiaction|[multilabel fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip)|todo|

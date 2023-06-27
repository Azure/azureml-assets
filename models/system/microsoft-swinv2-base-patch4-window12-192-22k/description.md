The Swin Transformer is a type of Vision Transformer used in both image classification and dense recognition tasks. It builds hierarchical feature maps by merging image patches in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window. Previous vision Transformers produce feature maps of a single low resolution and have quadratic computation complexity to input image size due to the computation of self-attention globally. Swin Transformer v2 has three main improvements which are a residual-post-norm method, a log-spaced continuous position bias method, and a self-supervised pre-training method called SimMIM. These improvements combined with cosine attention help improve training stability and reduce the need for vast labeled images.

> The above summary was generated using ChatGPT. Review the [original-model-card](https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

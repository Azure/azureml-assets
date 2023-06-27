The BEiT is a vision transformer that is similar to the BERT model, but is also capable of image analysis. The model is pre-trained on a large collection of images, and uses patches to analyze images. It uses relative position embeddings and mean-pooling to classify images, and can be used to extract image features for downstream tasks by placing a linear layer on top of the pre-trained encoder. You can place a linear layer on top of the [CLS] token or mean-pool the final hidden states of the patch embeddings, depending on the specifics of your task.

> The above summary was generated using ChatGPT. Review the [original-model-card](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

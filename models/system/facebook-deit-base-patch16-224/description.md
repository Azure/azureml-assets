This model is actually a more efficiently trained Vision Transformer (ViT). The Vision Transformer (ViT) is a transformer encoder model that is pre-trained and fine-tuned on a large collection of images in a supervised fashion. It is presented with images as sequences of fixed-size patches, which are linearly embedded, and before feeding the sequence to the layers of the Transformer encoder, absolute position embeddings are added. By pre-training the model, it is able to generate an inner representation of images that can be used to extract useful features for downstream tasks. For example, if one has a dataset of labeled images, a standard classifier can be trained by placing a linear layer on top of the pre-trained encoder. The last hidden state of the [CLS] token can be used as a representation of the entire image.

> The above summary was generated using ChatGPT. Review the [original-model-card](https://huggingface.co/facebook/deit-base-patch16-224) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

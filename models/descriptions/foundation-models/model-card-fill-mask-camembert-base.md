CamemBERT is a state-of-the-art language model for French developed by a team of researchers. It is based on the RoBERTa model and is available in 6 different versions on Hugging Face. It can be used for fill-in-the-blank tasks. However, it has been pretrained on a subcorpus of OSCAR which may contain lower quality data and personal and sensitive information. Also, there may be biases and historical stereotypes present in the model. The model is licensed under the MIT license, and more information can be found in the research paper and on the Camembert website. It was trained on the OSCAR dataset, which is a multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the Ungoliant architecture.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/camembert-base) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/azureml-sdk-fill-mask-online-endpoint)|[fill-mask-online-endpoint.sh](https://aka.ms/azureml-cli-fill-mask-online-endpoint)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://github.com/Azure/azureml-examples/tree/sitaram/finetunenotebooks/sdk/python/foundation-models/system/finetune/token-classification/token-classification.ipynb)|[token-classification.sh](https://github.com/Azure/azureml-examples/blob/sitaram/finetunenotebooks/cli/foundation-models/system/finetune/token-classification/token-classification.sh)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://github.com/Azure/azureml-examples/blob/sitaram/finetunenotebooks/cli/foundation-models/system/finetune/question-answering/extractive-qa.sh)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Fill Mask||[imdb](https://huggingface.co/datasets/imdb)|[evaluate-model-fill-mask.ipynb](https://aka.ms/azureml-eval-sdk-fill-mask/)|






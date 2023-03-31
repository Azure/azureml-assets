The BERT model is a pre-trained model that has been trained on a large corpus of English language data. The model was trained using a masked language modeling (MLM) objective, meaning that the model is able to predict words that were randomly masked in an input sentence. The BERT model can also predict if two sentences were originally consecutive in a text or not. This model is intended to be used as a tool for fine-tuning in various downstream tasks such as sequence classification and question answering, but is not recommended for tasks such as text generation. To use this model, you can utilize a pipeline specifically designed for masked language modeling.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/bert-base-cased) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
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


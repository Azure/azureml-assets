DistilGPT2 is a distilled version of GPT-2, which is a transformer-based language model with 124 million parameters and an English language license. It is intended to be used for similar uses with the increased functionality of being smaller and easier to run than the base model. DistilGPT2 was trained with knowledge distillation, following a procedure similar to the training procedure for DistilBERT. It has been evaluated on the WikiText-103 benchmark and has a perplexity of 21.1. Carbon emissions for DistilGPT2 are 149.2 kg eq. CO2.  The developers do not support use-cases that require the generated text to be true and recommend not using the model if the project could interact with humans without reducing bias first. It is recommended to check the OpenWebTextCorpus, OpenAI’s WebText dataset and Radford's research for further information about the training data and procedure. The Write With Transformers web app was built using DistilGPT2 and allows users to generate text directly from their browser.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/distilgpt2) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Text generation|Text generation|todo|[evaluate-model-text-generation.ipynb](https://aka.ms/azureml-eval-sdk-text-generation/)|

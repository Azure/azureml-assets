The RoBERTa Large model is a pretrained language model developed by the Hugging Face team, based on the transformer architecture. It was trained on a large corpus of English data in a self-supervised manner using the masked language modeling (MLM) objective. The model is case-sensitive and primarily intended for use in fine-tuning downstream tasks such as sequence classification, token classification, or question answering. It was trained on a combination of five datasets weighing 160GB of text, and uses a vocabulary size of 50,000 for tokenization. The model was trained for 500K steps on 1024 V100 GPUs with a batch size of 8K and a sequence length of 512. The optimizer used was Adam with a learning rate of 4e-4, β1=0.9, β2=0.98, and ϵ=1e-6, with a weight decay of 0.01 and learning rate warmup for 30,000 steps.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/roberta-large) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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






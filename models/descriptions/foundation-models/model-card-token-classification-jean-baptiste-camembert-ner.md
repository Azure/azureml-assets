Summary: camembert-ner is a NER model fine-tuned from camemBERT on the Wikiner-fr dataset and was validated on email/chat data. It shows better performance on entities that do not start with an uppercase. The model has four classes: O, MISC, PER, ORG and LOC. The model can be loaded using HuggingFace. The performance of the model is evaluated using seqeval. Overall, the model has precision 0.8859, recall 0.8971 and f1 0.8914. It shows good performance on PER entities, with precision, recall and f1 of 0.9372, 0.9598 and 0.9483 respectively. The model's author also provided a link to an article on how he used the model results to train a LSTM model for signature detection in emails.

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
|Token Classification||[CoNLL 2003](https://huggingface.co/datasets/conll2003)|[evaluate-model-token-classification.ipynb](https://aka.ms/azureml-eval-sdk-token-classification)|

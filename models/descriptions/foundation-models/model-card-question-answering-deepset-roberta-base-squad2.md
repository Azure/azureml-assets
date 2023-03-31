Roberta-base is a fine-tuned language model for extractive Question Answering in English, trained on the SQuAD2.0 dataset. It is based on the "roberta-base" model, developed by deepset and can be used with Haystack and Transformers. The model requires 4 Tesla v100s and has a batch size of 96, 2 epochs, and a learning rate of 3e-5. The model was evaluated on the SQuAD 2.0 dev set and achieved an exact match of 79.87 and an F1 score of 82.91. There is also a distilled version of this model available called "deepset/tinyroberta-squad2" which has a comparable prediction quality and runs twice as fast. Usage examples for the model are provided for Haystack and Transformers. The authors of the model are Branden Chan, Timo Möller, Malte Pietsch, and Tanay Soni from deepset.ai.

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
|Question Answering||[Squad v2](https://huggingface.co/datasets/squad_v2)|[evaluate-model-question-answering.ipynb](https://aka.ms/azureml-eval-sdk-question-answering)|**

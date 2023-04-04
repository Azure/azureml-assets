RoBERTa is a transformer-based language model that was fine-tuned from RoBERTa large model on Multi-Genre Natural Language Inference (MNLI) corpus for English. It can be used for zero-shot classification tasks and can be accessed from GitHub Repo. It is important to note that the model was trained on unfiltered data, so generated results may have disturbing and offensive stereotypes. Also, it should not be used to create hostile or alienating environments or to present factual or true representation of people or events.  The model is intended to be used on tasks that use the whole sentence or sequence classification, token classification or question answering. While it is pre-trained on a large corpus of English data, it's only intended to be fine-tuned further on specific tasks. The data the model was trained on includes 160GB of English text from various sources including Wikipedia and news articles. Pretraining was done using V100 GPUs for 500,000 steps using the masked language modeling objective. The training procedure has also been mentioned that it was done using Adam optimizer with a batch size of 8,000 and a sequence length of 512. Additionally, the model is a case-sensitive model, the tokenization and masking is done for the model pre-processing. A pipeline for masked language modeling can be used to directly use the model, but it is highly recommended to use the fine-tuned models available on the model hub for the task that interests you. The bias from the training data will also affect all fine-tuned versions of this model. 

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/roberta-base) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Fill Mask|Fill Mask|[imdb](https://huggingface.co/datasets/imdb)|[evaluate-model-fill-mask.ipynb](https://aka.ms/azureml-eval-sdk-fill-mask/)|


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://aka.ms/azureml-ft-cli-extractive-qa)

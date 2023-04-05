DeBERTa is an improvement of BERT and RoBERTa using disentangled attention and enhanced mask decoder. With 80GB training data, it outperforms BERT and RoBERTa on the majority of NLU tasks. The fine-tuned DeBERTa with MNLI task results in the best performance on SQuAD 1.1/2.0 and GLUE benchmark tasks. Further information is available in the official repository and the related paper.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/microsoft/deberta-large-mnli) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://aka.ms/azureml-ft-cli-extractive-qa)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Text Classification|Emotion Detection|[GoEmotions](https://huggingface.co/datasets/go_emotions)|[evaluate-model-text-classification.ipynb](https://aka.ms/azureml-eval-sdk-text-classification)|

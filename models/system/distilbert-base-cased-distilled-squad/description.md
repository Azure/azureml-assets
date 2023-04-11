The DistilBERT model is a distilled, smaller, faster, and cheaper version of the BERT model for Transformer-based language model. It is specifically trained for question answering in English and has been fine-tuned using knowledge distillation on SQuAD v1.1. It has 40% less parameters than bert-base-uncased and runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark. However, it is important to note that this model has some limitations and biases and should be used with caution. It should not be used to intentionally create hostile or alienating environments for people and it was not trained to be factual or true representations of people or events. The model has a F1 score of 87.1 on the SQuAD v1.1 dev set, but has unknown environmental impact as the carbon emissions and cloud provider are unknown.  The model was developed by Hugging Face and is licensed under Apache 2.0. The authors of the Model Card are the Hugging Face team.
The DistilBERT model requires significant computational power to train, as it was trained using 8 16GB V100 GPUs and for 90 hours. So it is important to keep that in mind when developing applications that utilize this model.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/distilbert-base-cased-distilled-squad) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[question-answering-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-question-answering)|[question-answering-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-question-answering)
Batch | coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://aka.ms/azureml-ft-cli-extractive-qa)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Question Answering|Extractive Q&A|[Squad v2](https://huggingface.co/datasets/squad_v2)|[evaluate-model-question-answering.ipynb](https://aka.ms/azureml-eval-sdk-question-answering)|**


#### Sample input
```json
{
    "inputs": {
        "question": ["What is my name?", "Where do I live?"],
        "context": ["My name is John and I live in Seattle.", "My name is Ravi and I live in Hyderabad."]
    }
}
```

#### Sample output
```json
[
    {
        "score": 0.9956907033920288,
        "start": 11,
        "end": 15,
        "answer": "John"
    },
    {
        "score": 0.9792415499687195,
        "start": 30,
        "end": 39,
        "answer": "Hyderabad"
    }
]
```

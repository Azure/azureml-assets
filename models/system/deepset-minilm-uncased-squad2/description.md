The MiniLM-L12-H384-uncased model is a microsoft language model for extractive question answering in English. It was trained on the SQuAD 2.0 dataset and has been evaluated on the SQuAD 2.0 dev set with the official eval script. The model's performance results were an exact match of 76.13 and F1 score of 79.50. The model can be used with Transformers, FARM, or haystack. The model was developed by Vaishali Pal, Branden Chan, Timo Möller, Malte Pietsch, and Tanay Soni, who are employees of deepset, a company focused on bringing NLP to the industry via open source.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/deepset/minilm-uncased-squad2) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
|Question Answering|Extractive Q&A|[Squad v2](https://huggingface.co/datasets/squad_v2)|[evaluate-model-question-answering.ipynb](https://aka.ms/azureml-eval-sdk-question-answering)|


### Sample inputs and outputs (for real-time inference)

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
        "score": 0.9982209801673889,
        "start": 11,
        "end": 15,
        "answer": "John"
    },
    {
        "score": 0.9689329266548157,
        "start": 30,
        "end": 39,
        "answer": "Hyderabad"
    }
]
```

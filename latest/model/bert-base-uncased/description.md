BERT is a pre-trained model in the field of NLP (natural language processing) released by Google. It is an AI language model that has been trained on a large corpus of English data using a self-supervised method, learning to predict masked words in a sentence and to predict if two sentences are consecutive or not. This model comes in several variations, including an English "uncased" version, Chinese and multilingual cased/uncased versions, and 24 smaller models. It is primarily used to fine-tune on downstream tasks for NLP, such as sequence classification, token classification, or question answering, although it can also be used for masked language modeling and next sentence prediction. The model was trained on BookCorpus and English Wikipedia data and its training procedure involved preprocessing the data, tokenizing it, and masking 15% of the tokens. The BERT model is fine-tuned on various tasks, and its test results on these tasks have shown impressive accuracy. 

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/bert-base-uncased) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-fill-mask)|[fill-mask-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-fill-mask)
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
|Fill Mask|Fill Mask|[imdb](https://huggingface.co/datasets/imdb)|[evaluate-model-fill-mask.ipynb](https://aka.ms/azureml-eval-sdk-fill-mask/)|


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["Paris is the [MASK] of France.", "Today is a [MASK] day!"]
    },
    "parameters": {
        "top_k": 2
    }
}
```

#### Sample output
```json
[
    [
        {
            "score": 0.9969369173049927,
            "token": 3007,
            "token_str": "capital",
            "sequence": "paris is the capital of france."
        },
        {
            "score": 0.000591485935728997,
            "token": 2540,
            "token_str": "heart",
            "sequence": "paris is the heart of france."
        }
    ],
    [
        {
            "score": 0.20859259366989136,
            "token": 2502,
            "token_str": "big",
            "sequence": "today is a big day!"
        },
        {
            "score": 0.17938676476478577,
            "token": 2307,
            "token_str": "great",
            "sequence": "today is a great day!"
        }
    ]
]
```

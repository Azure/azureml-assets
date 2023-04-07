The DistilBERT model is a smaller, faster version of the BERT model for Transformer-based language modeling with 40% fewer parameters and 60% faster run time while retaining 95% of BERT's performance on the GLUE language understanding benchmark. This English language question answering model has a F1 score of 87.1 on SQuAD v1.1 and was developed by Hugging Face under the Apache 2.0 license. Training the model requires significant computational power, such as 8 16GB V100 GPUs and 90 hours. Intended uses include fine-tuning on downstream tasks, but it should not be used to create hostile or alienating environments and limitations and biases should be taken into account.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/distilbert-base-cased) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-fill-mask)|[fill-mask-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-fill-mask)
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
            "score": 0.6490016579627991,
            "token": 2364,
            "token_str": "capital",
            "sequence": "Paris is the capital of France."
        },
        {
            "score": 0.02586420811712742,
            "token": 1946,
            "token_str": "seat",
            "sequence": "Paris is the seat of France."
        }
    ],
    [
        {
            "score": 0.0865759551525116,
            "token": 7310,
            "token_str": "wonderful",
            "sequence": "Today is a wonderful day!"
        },
        {
            "score": 0.08498429507017136,
            "token": 2712,
            "token_str": "beautiful",
            "sequence": "Today is a beautiful day!"
        }
    ]
]
```

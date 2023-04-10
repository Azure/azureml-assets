CamemBERT is a state-of-the-art language model for French developed by a team of researchers. It is based on the RoBERTa model and is available in 6 different versions on Hugging Face. It can be used for fill-in-the-blank tasks. However, it has been pretrained on a subcorpus of OSCAR which may contain lower quality data and personal and sensitive information. Also, there may be biases and historical stereotypes present in the model. The model is licensed under the MIT license, and more information can be found in the research paper and on the Camembert website. It was trained on the OSCAR dataset, which is a multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the Ungoliant architecture.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/camembert-base) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
        "input_string": ["Paris is the <mask> of France.", "Today is a <mask> day!"]
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
            "score": 0.38382619619369507,
            "token": 23151,
            "token_str": "city",
            "sequence": "Paris is the city of France."
        },
        {
            "score": 0.1262788623571396,
            "token": 6383,
            "token_str": "City",
            "sequence": "Paris is the City of France."
        }
    ],
    [
        {
            "score": 0.1769799441099167,
            "token": 24041,
            "token_str": "great",
            "sequence": "Today is a great day!"
        },
        {
            "score": 0.16837772727012634,
            "token": 7332,
            "token_str": "new",
            "sequence": "Today is a new day!"
        }
    ]
]
```

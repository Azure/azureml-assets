DeBERTa is a model that improves on the BERT and RoBERTa models by using disentangled attention and an enhanced mask decoder. It performance better on several NLU tasks than RoBERTa with 80GB training data. The DeBERTa XLarge model has 48 layers and a hidden size of 1024 with 750 million parameters. It demonstrates good results when fine-tuned on several NLU tasks like SQuAD and GLUE benchmark. If you use DeBERTa in your work, the authors request that you cite their papers.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/microsoft/deberta-xlarge) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
            "score": 0.00045154482359066606,
            "token": 22569,
            "token_str": " Tig",
            "sequence": "Paris is the Tig of France."
        },
        {
            "score": 0.00027906251489184797,
            "token": 23408,
            "token_str": " promoter",
            "sequence": "Paris is the promoter of France."
        }
    ],
    [
        {
            "score": 0.0003339046670589596,
            "token": 13093,
            "token_str": "edd",
            "sequence": "Today is aedd day!"
        },
        {
            "score": 0.0003173339064233005,
            "token": 42781,
            "token_str": " hallucinations",
            "sequence": "Today is a hallucinations day!"
        }
    ]
]
```

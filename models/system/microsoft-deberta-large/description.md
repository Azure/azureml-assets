Decoding-enhanced BERT with Disentangled Attention is that it is an improvement of the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With 80GB training data, it outperforms the BERT and RoBERTa models in many Natural Language Understanding (NLU) tasks. Key results can be found on the SQuAD 1.1/2.0 and GLUE benchmark tasks when fine-tuned with the MNLI task. The details are available in the official repository and a related paper. If it's useful, cite the paper as described in the citation.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/microsoft/deberta-large) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|coming soon


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
            "score": 0.0003298197698313743,
            "token": 38529,
            "token_str": " circum",
            "sequence": "Paris is the circum of France."
        },
        {
            "score": 0.0003219878126401454,
            "token": 44324,
            "token_str": " sidel",
            "sequence": "Paris is the sidel of France."
        }
    ],
    [
        {
            "score": 0.0004163646372035146,
            "token": 29992,
            "token_str": " Ou",
            "sequence": "Today is a Ou day!"
        },
        {
            "score": 0.00037583630182780325,
            "token": 4987,
            "token_str": "uz",
            "sequence": "Today is auz day!"
        }
    ]
]
```

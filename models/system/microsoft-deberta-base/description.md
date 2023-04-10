DeBERTa is a version of the BERT model that has been improved through the use of disentangled attention and enhanced mask decoders. It outperforms BERT and RoBERTa on a majority of NLU tasks using 80GB of training data. It has been fine-tuned on NLU tasks and has achieved dev results on SQuAD 1.1/2.0 and MNLI tasks.If you find the model useful please cite the paper. 
Please check the official repository for more detailed updates.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/microsoft/deberta-base) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
            "score": 0.0014589033089578152,
            "token": 36858,
            "token_str": " Hamb",
            "sequence": "Paris is the Hamb of France."
        },
        {
            "score": 0.0012754832860082388,
            "token": 46353,
            "token_str": "…", 
            "sequence": "Paris is the … of France."
        }
    ],
    [
        {
            "score": 0.0020057905931025743,
            "token": 47818,
            "token_str": "gently",
            "sequence": "Today is agently day!"
        },
        {
            "score": 0.0012056897394359112,
            "token": 32503,
            "token_str": "cand",
            "sequence": "Today is acand day!"
        }
    ]
]
```

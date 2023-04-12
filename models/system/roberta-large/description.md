The RoBERTa Large model is a pretrained language model developed by the Hugging Face team, based on the transformer architecture. It was trained on a large corpus of English data in a self-supervised manner using the masked language modeling (MLM) objective. The model is case-sensitive and primarily intended for use in fine-tuning downstream tasks such as sequence classification, token classification, or question answering. It was trained on a combination of five datasets weighing 160GB of text, and uses a vocabulary size of 50,000 for tokenization. The model was trained for 500K steps on 1024 V100 GPUs with a batch size of 8K and a sequence length of 512. The optimizer used was Adam with a learning rate of 4e-4, β1=0.9, β2=0.98, and ϵ=1e-6, with a weight decay of 0.01 and learning rate warmup for 30,000 steps.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/roberta-large) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

| Task      | Use case  | Dataset                                      | Python sample (Notebook)                                                     | CLI with YAML                                                              |
|-----------|-----------|----------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Fill Mask | Fill Mask | [imdb](https://huggingface.co/datasets/imdb) | [evaluate-model-fill-mask.ipynb](https://aka.ms/azureml-eval-sdk-fill-mask/) | [evaluate-model-fill-mask.yml](https://aka.ms/azureml-eval-cli-fill-mask/) |


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
            "score": 0.9945607781410217,
            "token": 812,
            "token_str": " capital",
            "sequence": "Paris is the capital of France."
        },
        {
            "score": 0.0038944401312619448,
            "token": 1867,
            "token_str": " Capital",
            "sequence": "Paris is the Capital of France."
        }
    ],
    [
        {
            "score": 0.231650248169899,
            "token": 372,
            "token_str": " great",
            "sequence": "Today is a great day!"
        },
        {
            "score": 0.15877646207809448,
            "token": 2721,
            "token_str": " beautiful",
            "sequence": "Today is a beautiful day!"
        }
    ]
]
```

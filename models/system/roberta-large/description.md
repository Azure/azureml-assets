The RoBERTa Large model is a pretrained language model developed by the Hugging Face team, based on the transformer architecture. It was trained on a large corpus of English data in a self-supervised manner using the masked language modeling (MLM) objective. The model is case-sensitive and primarily intended for use in fine-tuning downstream tasks such as sequence classification, token classification, or question answering. It was trained on a combination of five datasets weighing 160GB of text, and uses a vocabulary size of 50,000 for tokenization. The model was trained for 500K steps on 1024 V100 GPUs with a batch size of 8K and a sequence length of 512. The optimizer used was Adam with a learning rate of 4e-4, β1=0.9, β2=0.98, and ϵ=1e-6, with a weight decay of 0.01 and learning rate warmup for 30,000 steps.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/roberta-large" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-fill-mask" target="_blank">fill-mask-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-fill-mask" target="_blank">fill-mask-online-endpoint.sh</a>
Batch | coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Token Classification|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">token-classification.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">token-classification.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Fill Mask|Fill Mask|<a href="https://huggingface.co/datasets/imdb" target="_blank">imdb</a>|<a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a>|


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

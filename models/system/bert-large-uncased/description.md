BERT is a pre-trained language model created by the Hugging Face team that uses masked language modeling (MLM) on a large corpus of English data. Its primary uses are for sequence classification and question answering, and it is not intended for text generation. It is important to note that this particular BERT model is cased, making a distinction between 'english' and 'English'. The model can be fine-tuned for downstream tasks, and it is described as having 24 layers, 1024 hidden dimensions, 16 attention heads, and 336M parameters. It's most effective on use cases that involve using the entire sentence to make decisions, whereas tasks such as text-generations, you should use GPT2.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/bert-large-uncased" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

| Task      | Use case  | Dataset                                      | Python sample (Notebook)                                                     | CLI with YAML                                                              |
|-----------|-----------|----------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Fill Mask | Fill Mask | <a href="https://huggingface.co/datasets/imdb" target="_blank">imdb</a> | <a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a> |


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
            "score": 0.9984765648841858,
            "token": 3007,
            "token_str": "capital",
            "sequence": "paris is the capital of france."
        },
        {
            "score": 0.0005300433840602636,
            "token": 2415,
            "token_str": "center",
            "sequence": "paris is the center of france."
        }
    ],
    [
        {
            "score": 0.4392628073692322,
            "token": 2204,
            "token_str": "good",
            "sequence": "today is a good day!"
        },
        {
            "score": 0.09155428409576416,
            "token": 3376,
            "token_str": "beautiful",
            "sequence": "today is a beautiful day!"
        }
    ]
]
```

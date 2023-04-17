CamemBERT is a state-of-the-art language model for French developed by a team of researchers. It is based on the RoBERTa model and is available in 6 different versions on Hugging Face. It can be used for fill-in-the-blank tasks. However, it has been pretrained on a subcorpus of OSCAR which may contain lower quality data and personal and sensitive information. Also, there may be biases and historical stereotypes present in the model. The model is licensed under the MIT license, and more information can be found in the research paper and on the Camembert website. It was trained on the OSCAR dataset, which is a multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the Ungoliant architecture.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/camembert-base" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-fill-mask" target="_blank">fill-mask-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-fill-mask" target="_blank">fill-mask-online-endpoint.sh</a>
Batch | coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Token Classification|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">token-classification.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">token-classification.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>


### Model Evaluation

Task|Use case|Python sample (Notebook)|CLI with YAML
|--|--|--|--|
Fill Mask|Fill Mask|<a href="https://huggingface.co/datasets/imdb" target="_blank">imdb</a>|<a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


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

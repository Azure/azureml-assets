DistilRoBERTa base is a distilled version of the RoBERTa-base model, with 6 layers, 768 dimension, and 12 heads, and 82M parameters, it is faster than RoBERTa-base. The model is primarily intended for fine-tuning on whole sentence-based tasks such as sequence classification, token classification, and question answering but should not be used to generate harmful or alienating content. There is a risk of bias in the generated predictions as it may include harmful stereotypes. It is developed by Victor Sanh, Lysandre Debut, Julien Chaumond and Thomas Wolf of Hugging Face and is licensed under Apache 2.0. Users are encouraged to check out the RoBERTa-base model card for more information. When getting started with the model, it is recommended to use fine-tuned versions on the task that interests you.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/distilroberta-base" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

Task| Use case| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Fill Mask | Fill Mask | <a href="https://huggingface.co/datasets/imdb" target="_blank">imdb</a> | <a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


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
            "score": 0.6790168285369873,
            "token": 812,
            "token_str": " capital",
            "sequence": "Paris is the capital of France."
        },
        {
            "score": 0.051780153065919876,
            "token": 32357,
            "token_str": " birthplace",
            "sequence": "Paris is the birthplace of France."
        }
    ],
    [
        {
            "score": 0.20368757843971252,
            "token": 3610,
            "token_str": " busy",
            "sequence": "Today is a busy day!"
        },
        {
            "score": 0.09197165817022324,
            "token": 2721,
            "token_str": " beautiful",
            "sequence": "Today is a beautiful day!"
        }
    ]
]
```

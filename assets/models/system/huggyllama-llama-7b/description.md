
# Description
This repository provides the LLaMA-7b model weights, which are subject to a non-commercial license (refer to the LICENSE file). Access to this repository is granted only to individuals who have filled out a form to gain access to the model, specifically if they lost their model weights or encountered difficulties when converting them to the Transformers format.


# Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>


# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


## Sample inputs and outputs (for real-time inference)

### Sample input

```json
{
    "input_data": {
        "input_string": [
            "def fibonacci("
        ],
        "parameters": {
            "top_p": 0.9,
            "temperature": 0.6,
            "do_sample": true,
            "max_new_tokens": 50
        }
    }
}
```

### Sample output
```json
[
    {
        "0": "I believe the meaning of life is to learn to love.\\nI believe in a world of compassion, a world where love rules.\\nI believe in a world where people care for one another.\\nI believe in a world where people help each other.\\nI believe in a world where people are kind to each other.\\nI believe in a world where people are happy.\\nI believe in a world where people are peaceful.\\nI believe in a world where people are loving."
    }
]
```

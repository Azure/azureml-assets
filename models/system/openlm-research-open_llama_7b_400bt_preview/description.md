The OpenLLaMA project has released a public preview of their permissively licensed open source reproduction of Meta AI's LLaMA large language model. The preview includes a 7B and 3B variant of the OpenLLaMA model, trained on 700 billion and 600 billion tokens respectively. The weights can be used as a drop-in replacement for LLaMA 7B in existing implementations. The models were trained on the RedPajama dataset and evaluated on a wide range of tasks using lm-evaluation-harness. The results show that OpenLLaMA exhibits comparable performance to the original LLaMA and GPT-J across a majority of tasks, and outperforms them in some tasks. The OpenLLaMA project is currently focused on completing the training process on the entire RedPajama dataset.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/openlm-research/open_llama_7b_400bt_preview" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>

### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["My name is John and I am", "Once upon a time,"]
    }
}
```

#### Sample output
```json
[
    {
        "0": "My name is John and I am part of the world's largest open computer lab, one of the largest academic computer labs in the world, with over"
    },
    {
        "0": "Once upon a time, when I was twenty myself, I read John Milton's Paradise Lost in a small magazine. I was struck by a passage:"
    }
]
```
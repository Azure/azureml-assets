Databricks-dolly-v2-3b is a large language model that can perform various tasks based on instructions. It is trained on the Databricks machine learning platform, which allows you to use it for commercial purposes. It is based on the Pythia-2.8b model from EleutherAI and fine-tuned on a dataset of instructions and responses created by Databricks employees. The dataset covers different domains such as brainstorming, classification, closed QA, generation, information extraction, open QA and summarization.

Databricks-dolly-v2-3b is the smallest model in the Dolly v2 family, with 2.8 billion parameters. It is not the most advanced model, but it still shows impressive performance in following instructions that are not typical of the base model. It can generate text, answer questions, summarize articles, extract information, classify sentences, and more.

Dolly v2 also has two larger and more powerful models:

- Databricks-dolly-v2-12b, with 12 billion parameters and based on Pythia-12b
- Databricks-dolly-v2-7b, with 6.9 billion parameters and based on Pythia-6.9b

Source: Conversation with Bing, 1/9/2023
(1) databricks/dolly-v2-3b· Hugging Face. https://huggingface.co/databricks/dolly-v2-3b.
(2) GitHub - databrickslabs/dolly: Databricks’ Dolly, a large language .... https://github.com/databrickslabs/dolly.



> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/databricks/dolly-v2-3b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon



### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>


### Sample inputs and outputs (for real-time inference)

```json
{
    "input_data": {
        "input_string": ["My name is John and I am", "Once upon a time,"]
    }
}
```

#### Sample output
```json
[
    {
        "0": "My name is John and I am a student at UC Berkeley. It is my main interest to do research in the humanities. I am going to share"
    },
    {
        "0": "Once upon a time, they were just another small family, only three. She says one day that her father was getting a new license"
    }
]
```

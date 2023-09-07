GPT-2 is a transformer-based language model intended for AI researchers and practitioners. It was trained on unfiltered content from Reddit and may have biases. It is best used for text generation, but the training data has not been publicly released. It has several limitations and should be used with caution in situations that require truth and in systems that interact with humans. There are different versions of the model, including GPT-Large, GPT-Medium, and GPT-Xl, available for different use cases. The information provided by the OpenAI team is to complete and give specific examples of bias in their model.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/gpt2" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
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

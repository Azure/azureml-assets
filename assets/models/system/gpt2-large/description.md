The OpenAI GPT-2 is a language model that is intended to be used primarily by AI researchers and practitioners. It is capable of performing various uses, including writing assistance and creative writing, but is not recommended to be deployed in human interaction systems without a thorough study of its biases. The training data used to create this model was scraped from Reddit, excluding all pages of Wikipedia, and has not been publicly released. The model was trained on a very large corpus of English data in a self-supervised fashion, meaning it was pretrained on raw texts without human labeling. The evaluation information for this model comes from its associated paper and is evaluated on various language model benchmarks. The results are reported using invertible de-tokenizers to remove pre-processing artifacts.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/gpt2-large" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
        "0": "My name is John and I am a very good cook. My specialty is lasagna. I am not your typical lasagna producer. My wife and"
    },
    {
        "0": "Once upon a time, everyone believed that you had to be a member of the priesthood to be worthy of the blessings of salvation in the next life."
    }
]
```

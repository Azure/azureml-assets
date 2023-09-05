DistilGPT2 is a distilled version of GPT-2, which is a transformer-based language model with 124 million parameters and an English language license. It is intended to be used for similar uses with the increased functionality of being smaller and easier to run than the base model. DistilGPT2 was trained with knowledge distillation, following a procedure similar to the training procedure for DistilBERT. It has been evaluated on the WikiText-103 benchmark and has a perplexity of 21.1. Carbon emissions for DistilGPT2 are 149.2 kg eq. CO2.  The developers do not support use-cases that require the generated text to be true and recommend not using the model if the project could interact with humans without reducing bias first. It is recommended to check the OpenWebTextCorpus, OpenAI’s WebText dataset and Radford's research for further information about the training data and procedure. The Write With Transformers web app was built using DistilGPT2 and allows users to generate text directly from their browser.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/distilgpt2" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
        "0": "My name is John and I am the first person to ever make the same kind of a film. I've always been obsessed with the film, and"
    },
    {
        "0": "Once upon a time, though, we were always a different people than any other society. Many of us now live in one-of-kind communities"
    }
]
```

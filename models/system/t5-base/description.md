T5 Base is a text-to-text transformer model that can be used for a variety of NLP tasks, such as machine translation, document summarization, question answering and classification tasks, such as sentiment analysis. It was developed by a team at Google and is pre-trained on the Colossal Clean Crawled Corpus. It is licensed under Apache 2.0 and you can start using the model by installing the T5 tokenizer and model and following the examples provided in the Colab Notebook created by its developers. Be mindful of bias, risks and limitations that may arise while using this model.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/t5-base" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-translation" target="_blank">translation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-translation" target="_blank">translation-online-endpoint.sh</a>
Batch | coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Summarization|Summarization|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">CNN DailyMail</a>|<a href="https://aka.ms/azureml-ft-sdk-news-summary" target="_blank">news-summary.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-news-summary" target="_blank">news-summary.sh</a>
Translation|Translation|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">WMT16</a>|<a href="https://aka.ms/azureml-ft-sdk-translation" target="_blank">translation.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-translation" target="_blank">translation.sh</a>


### Model Evaluation

| Task        | Use case    | Dataset                                                            | Python sample (Notebook)                                                        | CLI with YAML                                                                 |
|-------------|-------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Translation | Translation | <a href="https://huggingface.co/datasets/wmt19/viewer/cs-en/" target="_blank">wmt19/cs-en</a> | <a href="https://aka.ms/azureml-eval-sdk-translation" target="_blank">evaluate-model-translation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-translation" target="_blank">evaluate-model-translation.yml</a> |


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["My name is John and I live in Seattle", "Berlin is the capital of Germany."]
    }
}
```

#### Sample output
```json
[
    {
        "0": "Mein Name ist John und ich lebe in Seattle"
    },
    {
        "0": "Berlin ist die Hauptstadt Deutschlands."
    }
]
```

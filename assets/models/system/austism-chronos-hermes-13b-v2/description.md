The Chronos-Hermes-13b-v2, a fusion of Chronos-13b-v2 and Nous-Hermes-Llama2-13b, boasts an imaginative writing style reminiscent of Chronos while maintaining coherence and improved adherence to instructions. This hybrid model offers long and exceptionally prose-rich outputs, supporting a maximum context length of 4096. It combines the vivid storytelling elements of Chronos with enhanced coherency, resulting in a model capable of producing evocative storywriting while following a narrative with great precision. Unlike its predecessor, this mix significantly reduces the tendency to produce nonsensical babble, making it a powerful tool for creative and structured writing.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/Austism/chronos-hermes-13b-v2" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-translation" target="_blank">translation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-translation" target="_blank">translation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-translation" target="_blank">translation-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Summarization|News Summary|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">CNN DailyMail</a>|<a href="https://aka.ms/azureml-ft-sdk-news-summary" target="_blank">news-summary.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-news-summary" target="_blank">news-summary.sh</a>
Translation|Translate English to Romanian|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">WMT16</a>|<a href="https://aka.ms/azureml-ft-sdk-translation" target="_blank">translate-english-to-romanian.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-translation" target="_blank">translate-english-to-romanian.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Translation | Translation | <a href="https://huggingface.co/datasets/wmt16/viewer/ro-en/train" target="_blank">wmt16/ro-en</a> | <a href="https://aka.ms/azureml-eval-sdk-translation" target="_blank">evaluate-model-translation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-translation" target="_blank">evaluate-model-translation.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["My name is John and I live in Seattle", "Berlin is the capital of Germany."]
    },
    "parameters": {
        "task_type": "translation_en_to_fr"
    }
}
```

#### Sample output
```json
[
  {
    "0": "My name is John and I live in Seattle. I am a software engineer and I love to write"
  },
  {
    "0": "Berlin is the capital of Germany.\nThe city is known for its rich history, diverse culture,"
  }
]
```

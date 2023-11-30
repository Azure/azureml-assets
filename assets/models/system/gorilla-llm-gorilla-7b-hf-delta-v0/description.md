Gorilla, developed by UC Berkeley, introduces a groundbreaking approach enabling Large Language Models (LLMs) to utilize APIs by generating semantically- and syntactically-correct API calls in response to natural language queries. This innovation allows LLMs to accurately invoke over 1,600 APIs, with a focus on reducing hallucination. The project also introduces APIBench, a comprehensive collection of APIs designed for easy training. Gorilla offers a pre-trained model, gorilla-7b-hf-delta-v0, capable of utilizing Hugging Face APIs with a 0-shot fine-tuning approach. The model can understand prompts in natural language, making it versatile for various tasks. The training pipeline includes both standard fine-tuning and a novel retriever-aware approach. For those interested in contributing or having their APIs incorporated, Gorilla invites participation through Discord, pull requests, or email. The model, based on the transformer architecture, represents an auto-regressive language model and was last updated on May 27, 2023. For further details, the website, GitHub, and associated paper provide additional information.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/gorilla-llm/gorilla-7b-hf-delta-v0" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
    "input_data": {
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
    "0": "My name is John and I live in Seattle\");prespresprespresprespresprespresprespres"
  },
  {
    "0": "Berlin is the capital of Germany.?,includesincludesincludesincludesincludesincludesincludesincludesincludesincludesincludesincludes"
  }
]
```

# Description
This repository provides a Japanese GPT-NeoX model with 3.6 billion parameters.
The model was trained using code based on EleutherAI/gpt-neox.
It is a 36-layer, 2816-hidden-size transformer-based language model.

# Pre-training
The model was trained on around 312.5B tokens from Japanese CC-100, Japanese C4, and Japanese Wikipedia to optimize a traditional language modelling objective.
A final validation perplexity of 8.68 has been reached.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/rinna/japanese-gpt-neox-3.6b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model. Some of the content has been made available below.

## Model Series

| Variant         | Link                                                                   |
| ----------------| -----------------------------------------------------------------------|
| 3.6B PPO        | https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo    |
| 3.6B SFT-v2     | https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2 |
| 3.6B SFT        | https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft    |
| 3.6B pretrained | https://huggingface.co/rinna/japanese-gpt-neox-3.6b                    |

## Tokenization
Uses a sentencepiece-based tokenizer with a vocabulary size of 32,000.
Utilizes sentencepiece's byte fallback feature to decompose unknown text pieces into UTF-8 byte pieces.
Supports features like turning off the automatic addition of a dummy prefix and preserving extra whitespaces.
Set use_fast=False to ensure correct functionality.

### Authors
Tianyu Zhao and Kei Sawada

#### License
rinna/japanese-gpt-neox-3.6b is made available under The MIT license.

# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>

# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": [
            "What is your favourite condiment?",
            "Do you have mayonnaise recipes?"
        ],
        "parameters": {
            "max_new_tokens": 100,
            "do_sample": true,
            "return_full_text": false
        }
    }
}
```

#### Sample output
```json
[
  {
    "0": " These this has ready to spill it in to the sleep, the pieces and the poems although we think about the concert, the pieces and the poems although we think about. When we were sleep, our couch and said those watches to sp"
  },
  {
    "0": " 」 : 『あなたは私が探し求めていた人ですか?』、 『私はあなたのものなのです。 』 : 『あなたと会った時に、ずっと言おうと思っていたわ。 でも、言わずにはいられなかったの。 アダマスからのメッセージは、あなたは愛に値することを私が示したので、 それをずっと探し求めていたのよ。 』 : 『アダマス、私に愛に値する人は誰なのか"
  }
]
```

# **Model Details**

The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. 
Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks tested.

For full details of this model please read [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).

## Model Architecture

Mistral-7B-v0.1 is a transformer model, with the following architecture choices:
- Grouped-Query Attention
- Sliding-Window Attention
- Byte-fallback BPE tokenizer

Mistral 7B v0.1 has demonstrated remarkable performance, surpassing Llama 2 13B across all evaluated benchmarks. Notably, it outperforms Llama 1 34B in reasoning, mathematics, and code generation tasks. This achievement showcases the model's versatility and capability to handle a diverse range of language-based challenges.

## Notice

Mistral 7B is a pretrained base model and therefore does not have any moderation mechanisms.


# Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Generation|Summarization|<a href="https://huggingface.co/datasets/samsum" target="_blank">Samsum</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-generation/summarization_with_text_gen.ipynb" target="_blank">summarization_with_text_gen.ipynb</a>| <a href="https://github.com/Azure/azureml-examples/blob/main/cli/foundation-models/system/finetune/text-generation/text-generation.sh">text-generation.sh</a>
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-classification/emotion-detection-llama.ipynb" target="_blank">emotion-detection-llama.ipynb</a>| <a href="https://github.com/Azure/azureml-examples/blob/main/cli/foundation-models/system/finetune/text-classification/emotion-detection.sh">emotion-detection.sh</a>

# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


## **Sample inputs and outputs (for real-time inference)**

### **Sample input**
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

### **Sample output**
```json
[
  {
    "0": "\n\nMayonnaise - can't be beat.\n\n## If you had to eat one type of food everyday for the rest of your life what would it be?\n\nMango. I'm an avid fruit and vegetable eater.\n\n## What is your favourite fruit and/or vegetable?\n\nMango! I eat an acre of these a year, which is almost two pounds a day.\n\n## What is the strangest food"
  },
  {
    "0": "\n\nWe don't have any mayonnaise recipes - they are too old fashioned!\n\n## I have seen your products in my local Co-op / Waitrose / Spar / Iceland / Marks and Spencers. Where can I buy more?\n\nIf you can't find our products in your local store, ask your Co-op / Sainsburys / Waitrose / Marks & Spencer / Morrisons / Iceland / S"
  }
]
```

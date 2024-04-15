# **Model Details**

The Mixtral-8x7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. 
Mixtral-8x7B-v0.1 outperforms Llama 2 70B on most benchmarks with 6x faster inference.

For full details of this model please read [release blog post](https://mistral.ai/news/mixtral-of-experts/).

## Model Architecture

Mixtral-8x7B-v0.1 is a decoder-only model with 8 distinct groups or the "experts". At every layer, for every token, a router network chooses two of these experts to process the token and combine their output additively. Mixtral has 46.7B total parameters but only uses 12.9B parameters per token using this technique. This enables the model to perform with same speed and cost as 12.9B model.



# Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Generation|Summarization|<a href="https://huggingface.co/datasets/samsum" target="_blank">Samsum</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-generation/summarization_with_text_gen.ipynb" target="_blank">summarization_with_text_gen.ipynb</a>| <a href="https://github.com/Azure/azureml-examples/blob/main/cli/foundation-models/system/finetune/text-generation/text-generation.sh">text-generation.sh</a>

# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# **Sample inputs and outputs**

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

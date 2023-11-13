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
        "input_string": ["What is your favourite condiment?","Do you have mayonnaise recipes?"],
        "parameters": {
            "max_new_tokens":100, 
            "do_sample":true,
            "return_full_text": false
        }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "Worcestershire sauce. (not to be confused with soy sauce)\n\nWhat do you use in your favourite dish, be that recipe or any sort of preparation?\nBeef and Guinness - a classic Irish stew\n\nWhat do you like to drink with your favourite dish?\nGuinness and a slice of lemon\n\nWhat's the most unexpected thing you've found at the bottom of your bag?\nA pen and an iphone"
  },
  {
    "0": "User 7: This is my go to recipe. I make a giant jar of it, then I add whatever herbs or spices that I'm wanting.\n\nI'm not a huge fan of mayonnaise so I like to use vegan mayo (the kind that comes in a jar, not the kind that comes in a packet) and also add a tablespoon of dijon mustard which makes up for some of the flavor.\n\nMayonnaise is"
  }
]
```

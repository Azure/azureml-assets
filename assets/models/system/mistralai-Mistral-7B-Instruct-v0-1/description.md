# **Model Details**
The Mistral-7B-Instruct-v0.1 Large Language Model (LLM) is a fine-tuned version of the Mistral-7B-v0.1 generative text model, created using publicly available conversation datasets. To use this model, prompts should be surrounded by [INST] and [\INST] tokens, with the first instruction starting with a sentence ID and ending with an end-of-sentence token ID. This format is available as a chat template for easy usage. The model architecture is based on Mistral-7B-v0.1 and includes features like Grouped-Query Attention, Sliding-Window Attention, and Byte-fallback BPE tokenizer.

If you encounter a specific error when using the model, you may need to install the transformers library from source to resolve it, although this issue should be fixed in versions after transformers-v4.33.4.

It's important to note that the Mistral 7B Instruct model lacks moderation mechanisms, and efforts are being made to make it more suitable for environments requiring controlled outputs. For full details, you can refer to the release blog post provided.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.


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

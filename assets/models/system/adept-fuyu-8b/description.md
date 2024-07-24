###Fuyu-8B is a smaller version of the multimodal model that powers Adept AI’s product. It is available on HuggingFace and has a much simpler architecture and training procedure than other multi-modal models, making it easier to understand, scale, and deploy. Fuyu-8B is designed from the ground up for digital agents, so it can support arbitrary image resolutions, answer questions about graphs and diagrams, answer UI-based questions, and do fine-grained localization on screen images. It is also fast, with responses for large images taking less than 100 milliseconds. Despite being optimized for Adept AI’s use-case, it performs well at standard image understanding benchmarks such as visual question-answering and natural-image-captioning.

Please note that Fuyu-8B is a base model. You may need to fine-tune the model for specific use cases like verbose captioning or multimodal chat. In our experience, the model responds well to few-shotting and fine-tuning for a variety of use-cases.

###Below are the details for your perusal:

# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>

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

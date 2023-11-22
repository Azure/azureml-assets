The cyberagent/open-calm-7b release under the CC BY-SA 4.0 license and its use of the GPT-NeoX library. 
OpenCALM:This is a suite of decoder-only language models developed by CyberAgent, Inc. It is specifically designed for processing Japanese language data.
Model type: Transformer-based Language Model: The reference to being a "Transformer-based Language Model" indicates that OpenCALM is built on the Transformer architecture. Transformers have proven to be highly effective in various natural language processing tasks.
Library: GPT-NeoX: GPT-NeoX is mentioned as the underlying library for OpenCALM. GPT-NeoX is an extension of the GPT (Generative Pre-trained Transformer) architecture, designed for large-scale language modeling tasks. The "NeoX" extension might include improvements or modifications for specific use cases.

The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/cyberagent/open-calm-7b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
    "0": " An interesting compound is LSD-2.0 (Linguistic Domains of Nucleic A type 2 receptor) which is called LSH2. LSH2 is a piece (2) compound.The phrase \"Non-Non\" means it remains almost always understood. You can say, 'Non-Non', the phr"
  },
  {
    "0": " Could you prepare the soups for the soups you've reached? I like the arabic soups, pork soups, cabbage soups and feta soups.I suppose that I was having the South African today was home in honour and had delicious tastiest delicious restaurant in your country on f"
  }
]
```
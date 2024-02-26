# **Model Details**

Name: Suchinthana/llama2-product-chat
Llama2 Product Chat is a 7 billion parameter large language model (LLM) designed for product-related conversations. It requires 13.5 GB VRAM, is maintained by Suchinthana, and falls under the Llama2 license. The model uses the LlamaForCausalLM architecture, has a context length of 4096 tokens, and a maximum length of 4096 tokens. It operates with Transformers version 4.31.0, employs the LlamaTokenizer, and has a vocabulary size of 32,000. The model files total 13.5 GB, distributed across seven parts. The Torch data type used is float16, and the model's initializer range is set to 0.02.

# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# **Sample inputs and outputs**

### **Sample input**
```json
{
    "input_data":{
       "input_string":["the meaning of life is"],
       "parameters":{
             "temperature":0.5,
             "top_p":0.5,
             "max_new_tokens":100,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is to find your gift. You are here to make manifest the glory of God within you. It’s not just to survive, but to thrive and to do so with some passion, some compassion, some humor, and some style.\n– Maya Angelou\nMaya Angelou was an American poet, author, and civil rights activist. She was born on April 4, 1928, in St. Louis, Missouri, and passed away on"
  }
]
```

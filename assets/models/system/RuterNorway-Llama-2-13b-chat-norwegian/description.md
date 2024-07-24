# **Model Details**

Name: RuterNorway/Llama-2-13b-chat-norwegian
The Llama-2-13b-chat-norwegian model, a variant of Meta's Llama 2 13b Chat model, fine-tuned for Norwegian. The model is trained on a mix of Norwegian datasets and is intended for commercial and research use in Norwegian, particularly as an assistant-like chat.

The prompt template for Llama2 Chat is highlighted, emphasizing the model's guidelines for helpful, respectful, and unbiased responses. The model was developed at Ruter AI Lab in 2023 and underwent training using Norwegian alpaca data, 15k samples from Norwegian OpenOrca, and a small subset of custom instructional data.

The limitations of the model are outlined, clarifying its focus on tasks like summarization, question answering, and chat rather than extensive knowledge about Norway. The data used for training is machine-translated, and the model is released as-is, potentially requiring prompt tuning for optimal results.

The license for Llama 2 and a disclaimer regarding its usage and potential outputs are provided. Credits acknowledge the development at Ruter AI Lab.

The Norwegian version of the information (in Norwegian) repeats the key details, including the model's purpose, training data, prompt template, limitations, license, disclaimer, and credits.

Note: Llama-2-13b-chat-norwegian is a variant of Meta´s Llama 2 13b Chat model
https://huggingface.co/meta-llama, https://huggingface.co/meta-llama/Llama-2-13b-chat-hf, https://ruter.no/

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
             "max_new_tokens":10,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is to find your gift. The purpose of life is"
  }
]
```

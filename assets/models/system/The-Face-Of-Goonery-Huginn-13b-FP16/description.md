# **Model Details**

Name: The-Face-Of-Goonery/Huginn-13b-FP16
this is a language model named "The-Face-Of-Goonery/Huginn-13b-FP16." This model is described as a merge of various existing models, including hermes, beluga, airoboros, chronos, and limarp. It is emphasized that this model exhibits significantly better quality than the previous chronos-beluga merge.

Huginn is positioned as a general-purpose model, designed to possess extensive knowledge, logical reasoning capabilities, and the ability to accurately follow instructions. Additionally, the model is noted for its proficiency in maintaining prose and creativity, making it suitable for writing-oriented tasks. The versatility of Huginn is highlighted, indicating its usefulness in roleplays, regular chatbot interactions, and assistant-like functions.

Overall, the information suggests that Huginn is a high-quality, multifaceted language model that excels in various applications, particularly in scenarios requiring logical thinking, knowledge retention, and creative writing.

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

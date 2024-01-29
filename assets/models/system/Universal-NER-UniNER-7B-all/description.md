# **Model Details**

The development of Universal-NER/UniNER-7B-all, recognized as the top-performing UniNER model. The model is a result of targeted distillation from large language models (LLMs), specifically focusing on open Named Entity Recognition (NER). The research, conducted by Wenxuan Zhou, Sheng Zhang, Yu Gu, Muhao Chen, and Hoifung Poon, explores the application of mission-focused instruction tuning to train smaller student models capable of excelling in open information extraction.
resulting in UniversalNER, a highly effective and efficient model for open Named Entity Recognition, surpassing other models in terms of accuracy and parameter efficiency. The release of resources encourages future research in the field.

Note: Check our Paper for more information - https://arxiv.org/abs/2308.03279 

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
             "max_new_tokens":50,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is to find it.\nThe meaning of life is to find it.\nThe meaning of life is to find it.\nThe meaning of life is to find it.\nThe meaning of life is to find it.\nThe meaning of life is"
  }
]
```

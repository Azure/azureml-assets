# **Model Details**

Name: core42/jais-13b
Jais-13b is a 13 billion parameter pre-trained bilingual language model for Arabic and English. Developed by Inception, MBZUAI, and Cerebras Systems, it employs a transformer-based decoder-only architecture with SwiGLU non-linearity. The model is released under the Apache 2.0 license and has achieved state-of-the-art performance on a comprehensive Arabic test suite. Intended uses include research, commercial applications like chat assistants and customer service, benefiting academics, businesses, and developers targeting Arabic-speaking audiences. However, it should not be used for malicious purposes, handling sensitive information, making high-stakes decisions without human oversight, or assumed to have proficiency in languages beyond Arabic and English. Efforts to minimize biases have been made, but some bias may still be present. The model is trained on a diverse bilingual corpus from the web and other sources, with augmented Arabic data through translation from high-quality English resources. Users are encouraged to provide feedback for continuous improvement.

Paper : https://arxiv.org/abs/2308.16149

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
    "0": "the meaning of life is to find your gift and to give it away."
  }
]
```

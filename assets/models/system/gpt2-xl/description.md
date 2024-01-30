# **Model Details**

Model Name: GPT-2 XL
Developed by: OpenAI
Version: 1.5B parameters
Model Type: Transformer-based language model
Language(s): English
License: Modified MIT License
Key Features:

Parameter Size:

A version of GPT-2 with 1.5 billion parameters.
Pretraining Objective:

Pretrained on the English language.
Utilizes a causal language modeling (CLM) objective during training.
Development:

Developed by OpenAI.
Associated research paper and GitHub repository provide additional details and resources.
Licensing:

License Type: Modified MIT License.
Related Models:

Other GPT-2 Variants:
GPT-2
GPT-Medium
GPT-Large
Note:
GPT-2 XL is a transformer-based language model with a substantial parameter size of 1.5 billion. It has been pretrained on the English language using a causal language modeling objective. Developed by OpenAI, it belongs to the GPT-2 family, and related models include GPT-2, GPT-Medium, and GPT-Large. Additional details, research papers, and development resources can be found in the associated GitHub repository. The model is released under a Modified MIT License.

Resources for more information: Check out our blog post-https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


## **Sample inputs and outputs (for real-time inference)**

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
  "the meaning of life is to be found in the heart of the person, not in the words of
  the Bible. " The church's statement said the church would continue to support the LGBT
  community and "will not be silent in the face of hate and violence." "We will not be
  silent in the face of hate and violence, " the statement said. "We will stand with our
  LGBT brothers and sisters in the fight for equality." The statement also said the church would continue to support the LGBT community and"
]
```

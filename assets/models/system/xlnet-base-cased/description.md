# **Model Details**

Model Introduction:

XLNet is an unsupervised language representation learning method.
Based on a novel generalized permutation language modeling objective.
Employs Transformer-XL as the backbone model.
Demonstrates excellent performance for language tasks with long context.
Introduced in the paper "XLNet: Generalized Autoregressive Pretraining for Language Understanding" by Yang et al.
Model Description:

Utilizes a generalized permutation language modeling objective.
Employs Transformer-XL as the backbone model.
Achieves state-of-the-art (SOTA) results on various downstream language tasks.
Notable for strong performance in tasks involving long context.
Intended Uses & Limitations:

Primarily intended for fine-tuning on downstream tasks.
Best suited for tasks that involve using the entire sentence (potentially masked) for decision-making.
Common use cases include sequence classification, token classification, and question answering.
Check the model hub for fine-tuned versions tailored to specific tasks of interest.

Note- It was introduced in the paper XLNet: Generalized Autoregressive Pretraining for Language Understanding by Yang et al. https://arxiv.org/abs/1906.08237

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
    "0": "the meaning of life is not always clear. The meaning of life is not"
  }
]
```

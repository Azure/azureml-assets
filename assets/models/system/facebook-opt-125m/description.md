# **Model Details**

The OPT (Open Pretrained Transformers) model, introduced by Meta AI and released on May 3rd, 2022, is a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters. The primary goal is to facilitate responsible and reproducible large language model (LLM) research at scale, allowing more researchers to study the impact of these models. Fundamentally, OPT aims to match the performance and sizes of GPT-3 class models while incorporating the latest best practices in data collection and efficient training.

Key Information:

Model Description:

Predominantly pretrained with English text, with a small amount of non-English data from CommonCrawl.
Pretrained using a causal language modeling (CLM) objective, belonging to the same family of decoder-only models as GPT-3.
Evaluation follows GPT-3's prompts and overall experimental setup.
Intended Uses & Limitations:

Can be used for prompting evaluation of downstream tasks and text generation.
Fine-tuning is possible on downstream tasks using the CLM example.
The model can be used directly with a text generation pipeline.
Limitations and biases are acknowledged, influenced by unfiltered internet content in the training data.
How to Use:

Example code for text generation using the model is provided.
Default generation is deterministic, but top-k sampling can be enabled by setting do_sample to True.
Limitations and Bias:

Acknowledges limitations in bias, safety, and quality issues in terms of generation diversity and hallucination.
The training data's lack of neutrality due to unfiltered internet content contributes to biases in the model.
Bias and limitations are expected to affect all fine-tuned versions of the model.
Training Data:

The model was trained on a large corpus, a union of five filtered datasets, totaling 180B tokens or 800GB of data.
Datasets include BookCorpus, CC-Stories, The Pile, Pushshift.io Reddit, and CCNewsV2.
Validation split is made up of 200MB of pretraining data sampled proportionally from each dataset's size.
Collection Process:

The dataset was collected from the internet and underwent standard data processing algorithms, including the removal of repetitive/non-informative text.
Disclaimer:

The model card content is based on the official model card written by the OPT team, and it is available in Appendix D of the paper.
In summary, OPT is designed to provide access to large language models for research purposes, emphasizing responsible use and acknowledging potential biases and limitations associated with unfiltered internet content in the training data

Note- For More Information Of OPT : https://arxiv.org/pdf/2205.01068.pdf

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
  {
    "0": "the meaning of life is to be a good person.\nI'm not sure if you're being sarcastic or not.\nI'm not sure if you're being sarcastic or not.\nI'm not sure if you're being sarcastic or not.\nI'm not sure if you're being sarcastic or not.\nI'm not sure if you're being sarcastic or not.\nI'm not sure if you're being sarcastic or not.\nI'm not sure if you're being sarcastic or not.\nI'm"
  }
]
```

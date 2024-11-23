# **Model Details**

Name: NumbersStation/nsql-llama-2-7B
NSQL (Neural SQL) is a family of autoregressive open-source large foundation models (FMs) specifically crafted for SQL generation tasks. The newly introduced member, NSQL-Llama-2-7B, is based on Meta's original Llama-2 7B model. It undergoes further pre-training on a dataset of general SQL queries and fine-tuning on text-to-SQL pairs.

Training Data:

General SQL queries sourced from The Stack (1M training samples).
Labeled text-to-SQL pairs from over 20 public web sources, excluding Spider and GeoQuery datasets for evaluation.
Evaluation Data:
Models are evaluated on two text-to-SQL benchmarks: Spider and GeoQuery.

Training Procedure:
NSQL was trained using cross-entropy loss to maximize likelihood, with a focus on the SQL portion of text-to-SQL pairs during fine-tuning. Training utilized 80GB A100s, incorporating data and model parallelism. Pre-training spanned 3 epochs, followed by 10 epochs of fine-tuning.

Intended Use and Limitations:
The model is tailored for text-to-SQL generation tasks, particularly in constructing SELECT queries based on table schema and natural language prompts. Optimal performance is achieved when adhering to the specified prompt format.

NSQL-Llama-2-7B is a specialized model for SQL generation, exhibiting proficiency in handling text-to-SQL tasks with defined use cases and format constraints.


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
             "max_new_tokens":20,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is to live it, and the meaning of death is to die."
  }
]
```

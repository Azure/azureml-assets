# **Model Details**

Model Name: cerebras-Cerebras-GPT-111M

Cerebras-GPT 111M is part of the Cerebras-GPT family, released for research into large language model (LLM) scaling laws using open architectures and datasets. The family includes models with varying parameter sizes, from 111M to 13B. These models are designed to demonstrate the simplicity and scalability of training LLMs on the Cerebras software and hardware stack.

Training of Cerebras-GPT models follows Chinchilla scaling laws, optimizing compute with 20 tokens per model parameter. The models were trained on the Andromeda AI supercomputer using Cerebras' weight streaming technology, which disaggregates compute from model storage, allowing efficient scaling across nodes via data parallelism.

Cerebras systems for pre-training and fine-tuning are available in the cloud through the Cerebras Model Studio, with CS-2 compatible checkpoints in the Cerebras Model Zoo. The models are transformer-based language models with a GPT-3 style architecture, using the Byte Pair Encoding tokenizer, a vocabulary size of 50257, and a sequence length of 2048.

The training optimizer is AdamW with specific parameters, and positional encoding is learned. The models are trained on "The Pile" dataset in English. For more details, users can refer to the Dense Scaling Laws Paper for training procedures, configuration files, and usage details. The license for the models is Apache 2.0.

For questions about Cerebras-GPT models, users can join the Cerebras Discord community. The model family is available on Hugging Face, and related models can be explored in the Cerebras-GPT Models section.

Check out our Blog Post and arXiv paper!:
https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/

https://arxiv.org/abs/2304.03208

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
    "input_string": ["the meaning of life is"],
    "parameters": {
      "temperature": 0.5,
      "top_p": 0.5,
      "max_new_tokens": 100,
      "do_sample": true
    }
  }
}
```

### **Sample output**

```json
[
  {
    "0": "the meaning of life is that the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who is in the position of the person who"
  }
]
```

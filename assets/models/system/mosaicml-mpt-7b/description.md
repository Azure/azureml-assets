# **Model Details**

MPT-7B is a decoder-style transformer model pretrained on 1 trillion tokens of English text and code. Developed by MosaicML, it belongs to the MosaicPretrainedTransformer (MPT) family, featuring a modified transformer architecture optimized for efficient training and inference. Key features include performance-optimized layer implementations and the elimination of context length limits through Attention with Linear Biases (ALiBi) instead of positional embeddings.

Key Characteristics:

Training Data: Pretrained from scratch on 1 trillion tokens of English text and code.
Architecture: Modified transformer architecture with ALiBi, enabling efficient training and stable convergence.
Training Efficiency: Designed for high throughput efficiency and stable convergence.
Inference Efficiency: Supports efficient serving with standard HuggingFace pipelines and NVIDIA's FasterTransformer.
Codebase: Utilizes the MosaicML LLM codebase found in the llm-foundry repository.
Distinguishing Features:

Licensing: Allows for commercial use, unlike LLaMA.
Data Size: Trained on a large dataset (1T tokens) compared to other models like Pythia, OpenLLaMA, and StableLM.
Input Handling: Prepared to handle extremely long inputs, supporting up to 84k tokens with ALiBi.
Speed: Capable of fast training and inference using FlashAttention and FasterTransformer.
Training Code: Equipped with highly efficient open-source training code available in the llm-foundry repository.
Finetuned Models:

MPT-7B-StoryWriter-65k+: Designed for reading and writing fictional stories with super-long context lengths (up to 65k tokens). ALiBi allows extrapolation beyond 65k tokens, demonstrating generations as long as 80k tokens on a single A100-80GB GPU. License: Apache 2.0.
MPT-7B-Instruct: A model for short-form instruction following, finetuned on a dataset derived from Databricks Dolly-15k and Anthropic Helpful and Harmless (HH-RLHF) datasets. License: Apache 2.0.
MPT-7B-Chat: A chatbot-like model for dialogue generation, finetuned on ShareGPT-Vicuna, HC3, Alpaca, HH-RLHF, and Evol-Instruct datasets.
It's important to note that MPT-7B is licensed under Apache 2.0, and it offers unique advantages in terms of licensing, data size, input handling, and efficiency compared to other models.

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
    "0": "the meaning of life is to find your gift. the purpose of life is"
  }
]
```

# **Model Details**

Name: llm-jp-1.3b-v1.0
The llm-jp-1.3b-v1.0 repository, which offers large language models developed by LLM-jp, a collaborative project launched in Japan. It introduces different model variants, including instruction models and pre-trained models. Checkpoints are available in the Hugging Face Transformers format, with Megatron-DeepSpeed format models accessible in a separate location.

The required libraries and their versions for using the models are listed, including torch, transformers, tokenizers, and accelerate. The training details are outlined, covering both pre-training and instruction tuning. Pre-training involved the use of 96 A100 40GB GPUs and Megatron-DeepSpeed software, while instruction tuning utilized 8 A100 40GB GPUs with TRL, PEFT, and DeepSpeed software.

The tokenizer for the model is based on the huggingface/tokenizers Unigram byte-fallback model, with vocabulary entries converted from llm-jp-tokenizer v2.1 (50k). The README.md of llm-ja-tokenizer is referenced for details on the vocabulary construction procedure. The model uses the Hugging Face Fast Tokenizer with a Unigram byte-fallback model, requiring tokenizers version 0.14.0 or later. The training algorithm involves SentencePiece Unigram byte-fallback, and the vocabulary size is 50,570, encompassing Japanese, English, and source code.

Datasets used for pre-training are a blend of various datasets, providing a comprehensive overview of the llm-jp-1.3b-v1.0 repository and the associated language models.

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
    "0": "the meaning of life is to be able to share it with others. there are many ways to share the meaning of life. One of the most common ways is through the sharing of the word "love." The word""
  }
]
```

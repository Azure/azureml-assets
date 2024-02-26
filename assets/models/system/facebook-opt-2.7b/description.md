# **Model Details**

Model Name: opt-2.7b
OPT, introduced in a paper by Meta AI, is a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters. Released for researchers, OPT aims to facilitate reproducible and responsible research on large language models. Trained predominantly in English, OPT leverages datasets like BookCorpus, CC-Stories, The Pile, Pushshift.io Reddit, and CCNewsV2, accumulating 180B tokens. The dataset underwent filtering and preprocessing, removing repetitive content. OPT, belonging to the decoder-only model family like GPT-3, uses causal language modeling (CLM) for pretraining. Tokenization is done using GPT2 byte-level Byte Pair Encoding (BPE) with a vocabulary size of 50272 and inputs composed of 2048 consecutive tokens.

Resources for more information: please read the official paper. https://arxiv.org/abs/2205.01068

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
    "0": "the meaning of life is to have fun.\nI don't think that's the meaning of life. I think the meaning of life is to find out what the meaning of life is.\nI think the meaning of life is to find out what the meaning of life is.\nI think the meaning of life is to find out what the meaning of life is.\nI think the meaning of life is to find out what the meaning of life is.\nI think the meaning of life is to find out what the"
  }
]
```

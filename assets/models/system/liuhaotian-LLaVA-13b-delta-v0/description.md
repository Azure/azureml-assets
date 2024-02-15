# **Model Details**

Model Name: liuhaotian-LLaVA-13b-delta-v0
LLaVA is an open-source chatbot based on the transformer architecture, fine-tuned using LLaMA/Vicuna on GPT-generated multimodal instruction-following data. The model was trained in April 2023 and is intended for research on large multimodal models and chatbots. Its primary users are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

The model's training dataset includes 595K filtered image-text pairs from CC3M and 150K GPT-generated multimodal instruction-following data. For evaluation, the model is assessed using 90 visual reasoning questions created from 30 images randomly sampled from COCO val 2014. These questions cover conversational, detailed description, and complex reasoning types. GPT-4 is employed to judge the model outputs. Additionally, the model is evaluated on the ScienceQA dataset, achieving a new state-of-the-art performance in synergy with GPT-4.

NOTE: This "delta model" cannot be used directly.
Users have to apply it on top of the original LLaMA weights to get actual LLaVA weights.
See https://github.com/haotian-liu/LLaVA#llava-weights for instructions.

See for More Details: https://llava-vl.github.io/

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
      "max_new_tokens": 10,
      "do_sample": true
    }
  }
}
```

### **Sample output**

```json
[
  {
    "0": "the meaning of life isnost Gru<im_patch> vague Could DO Maybe Liv Depisation"
  }
]
```

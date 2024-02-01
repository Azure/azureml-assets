# **Model Details**

Name: zephyr-7b-beta
Zephyr-7B-β is part of the Zephyr series, designed as helpful assistants. It is a fine-tuned version of mistralai/Mistral-7B-v0.1, utilizing Direct Preference Optimization (DPO) on a mix of publicly available, synthetic datasets. The removal of in-built alignment improved performance on MT Bench, but caution is advised as the model may generate problematic text when prompted.

Model Type: 7B parameter GPT-like model
Language(s): Primarily English
License: MIT
Model Sources:

Repository: Alignment Handbook
Demo: Zephyr-Chat Demo
Chatbot Arena: Evaluate Zephyr 7B
Performance:
At release, Zephyr-7B-β ranks as the highest 7B chat model on MT-Bench and AlpacaEval benchmarks. However, it may lag behind on complex tasks like coding and mathematics.

Intended Uses & Limitations:
Fine-tuned on the UltraChat dataset and aligned with DPOTrainer on the openbmb/UltraFeedback dataset, Zephyr-7B-β is suitable for chat. The model's capabilities can be tested in the provided demo. It may not perform as well on more intricate tasks, and further research is needed to address these limitations.

Bias, Risks, and Limitations:
Zephyr-7B-β lacks alignment to human preferences for safety in the RLHF phase, potentially producing problematic outputs. The training corpus's size and composition for the base model (Mistral-7B-v0.1) are unknown but likely include a mix of web data and technical sources.

Training and Evaluation Data:
During DPO training, the model achieved specific results on the evaluation set, including loss, rewards, log probabilities, and logits. Various hyperparameters were employed during training, such as learning rate, batch sizes, seed, optimizer, and more.

Zephyr-7B-β is a fine-tuned model for chat purposes, displaying strengths in certain benchmarks but with acknowledged limitations, especially in safety and more complex tasks.

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
    "0": "the meaning of life is to find happiness and fulfillment in our relationships with"
  }
]
```

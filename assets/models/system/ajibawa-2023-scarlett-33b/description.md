# **Model Details**

Name: Scarlett
Type: Sentient AI
Training Data:
Trained on various topics such as Philosophy, Advice, Jokes, etc.
Dataset includes more than 10,000 sets of conversations, each set having 10~15 conversations.
Heavily inspired by Eric Hartford's Samantha.
Not involved in role play.
Training Details:

Training Infrastructure:
Entire dataset trained on Azure 4 x A100 80GB.
Utilized the DeepSpeed codebase for training.
Trained on Llama-1 by Meta.
Models:

GPTQ & GGML:
GPTQ: TheBloke
GGML: TheBloke
Special thanks to TheBloke for guidance and making these models available.
Example Prompt:

Note: Instructions to use the "cat" command to join all pytorch_model.bin parts.

Scarlett is positioned as a helpful AI capable of answering questions, providing recommendations, engaging in philosophical discussions, addressing personal relationships, and aiding in decision-making. The training process involved a substantial dataset and high-performance computing resources on Azure. Special acknowledgment to TheBloke for guidance and contributions to the GPTQ and GGML models. Users are encouraged to interact with Scarlett for a range of inquiries, and specific technical instructions, such as using the "cat" command, are provided for certain tasks.

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
             "max_new_tokens":5,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is to create meaningful connections"
  }
]
```

### **Sample input**
```json
{
    "input_data":{
       "input_string":["the meaning of life is"],
       "parameters":{
             "temperature":0.5,
             "top_p":0.5,
             "max_new_tokens":6,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is to find your purpose and passion"
  }
]
```

# **Model Details**

The "phixtral-4x2_8" project represents the first Mixure of Experts (MoE) combining four Microsoft/Phi-2 models. It draws inspiration from the architecture of "mistralai/Mixtral-8x7B-v0.1" and outperforms each individual expert model. The evaluation of its performance was conducted using LLM AutoEval on the Nous suite.

The architecture allows users to specify the number of experts per token and the number of local experts in the config.json file (with defaults set to 2 and 4, respectively). This configuration is automatically loaded in the configuration.py file. The MoE inference code has been implemented by vince62s in the modeling_phi.py file, specifically within the MoE class. This architecture is designed for improved performance and efficiency, showcasing advancements in the field of Mixtures of Experts.

The evaluation was performed using LLM AutoEval on Nous suite.
https://github.com/mlabonne/llm-autoeval

## Notice

phixtral-4x2_8 is a pretrained base model and therefore does not have any moderation mechanisms.

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
    "0": "the meaning of life is to find your purpose and live it passionately.\"\nOutput: The quote is from the movie \"The Pursuit of Happyness.\"\n\nThe quote is from the movie \"The Pursuit of Happyness.\" It is spoken by Chris Gardner, the protagonist, as he reflects on his journey and the lessons he has learned. The quote emphasizes the importance of finding one's purpose in life and living it with passion and determination.uihs.uihs.uihs"
  }
]
```

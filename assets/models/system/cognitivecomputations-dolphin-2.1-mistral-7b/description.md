# **Model Details**

Name: cognitivecomputations/dolphin-2.1-mistral-7b
Dolphin-2.1-mistral-7b's training was sponsored by https://a16z.com/supporting-the-open-source-ai-community/

The model based on mistralAI with an Apache-2.0 license, making it suitable for both commercial and non-commercial use. The model is explicitly mentioned to be uncensored, with the dataset filtered to remove alignment and bias, aiming for increased compliance. Users are advised to implement their own alignment layer before exposing the model as a service, and caution is urged due to the model's potential compliance with various requests, including unethical ones.

The dataset used is Dolphin, which is an open-source implementation of Microsoft's Orca. The dataset has been modified for uncensoring, deduplication, cleaning, and improved quality. Additionally, Jon Durbin's Airoboros dataset has been incorporated to enhance creativity.

Training details indicate that it took 48 hours to train the model for 4 epochs on 4x A100 GPUs. Gratitude is expressed towards the generous sponsorship from a16z that made the model possible. Microsoft is acknowledged for authoring the Orca paper that inspired this work. Special thanks are extended to individuals named Wing Lian and TheBloke for helpful advice, and huge appreciation is directed towards Wing Lian and the Axolotl contributors for providing what is described as the best training framework.

Lastly, users are reminded of their responsibility for any content created using this model and are encouraged to enjoy it responsibly. A link to a blog post about uncensored models by the model's creator, erichartford, is provided for further information.

Note: Dolphin 2.1 https://erichartford.com/dolphin

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
    "0": "the meaning of life is to be found in the pursuit of happiness. "
  }
]
```

# **Model Details**

Name: HuggingFaceH4/starchat-beta
StarChat is a series of language models designed as helpful coding assistants. StarChat-β, the second model in this series, is a fine-tuned version of StarCoderPlus. The fine-tuning utilizes an "uncensored" variant of the openassistant-guanaco dataset, enhancing performance on coding tasks. Note that this model may produce problematic outputs when prompted.

Model Details:

Model type: A 16B parameter GPT-like model fine-tuned on an "uncensored" variant of the openassistant-guanaco dataset.
Language(s): Primarily English and supports 80+ programming languages.
License: BigCode Open RAIL-M v1
Finetuned from: bigcode/starcoderplus
Model Sources:

Intended Uses & Limitations:
Intended for chat and coding assistance.
Caution advised for potential problematic outputs.
Bias and demographic skewing from GitHub community demographics.
May produce syntactically valid but semantically incorrect code.
Potential for generating false URLs that require careful inspection.
Training and Evaluation:

Trained on "uncensored" openassistant-guanaco dataset.
Utilizes the ShareGPT filtering recipe behind the WizardLM.
Hyperparameters, including learning rate, batch size, and others, specified for training.
In summary, StarChat-β serves as a coding assistant, trained and fine-tuned for code-related tasks, offering capabilities in multiple programming languages. Users are cautioned about potential problematic outputs and advised to use the model responsibly for educational and research purposes.

Repository: https://github.com/bigcode-project/starcoder
Demo: https://huggingface.co/spaces/HuggingFaceH4/starchat-playground

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
    "0": "the meaning of life is to find your gift. the purpose of life is"
  }
]
```

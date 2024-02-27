# **Model Details**

Name: Llama-2-7b-chat-hf
Llama 2, a collection of pretrained and fine-tuned generative text models, with variations ranging from 7 billion to 70 billion parameters. The 7B model is highlighted, optimized for dialogue use cases and converted for Hugging Face Transformers format. Model details, including developers (Meta), variations, input/output formats, and architecture (optimized transformer) are provided.

The training data, parameters, content length, tokens, and learning rate for different Llama 2 variations are listed. The models were trained between January 2023 and July 2023. The status indicates that this is a static model trained offline, with future releases planned for improved safety based on community feedback.

The license for Llama 2 is specified as a custom commercial license available on Meta's website. Intended use cases include commercial and research applications in English, specifically for assistant-like chat. Out-of-scope uses, such as violating laws or using languages other than English, are mentioned.

Hardware and software details cover training factors, carbon footprint considerations, and energy consumption during pretraining. The text provides information on GPU hours, power consumption, and carbon emissions for Llama 2 models, with Meta offsetting 100% of the emissions through a sustainability program.

The training data overview mentions pretraining on 2 trillion tokens from publicly available sources, fine-tuning with instruction datasets, and over a million human-annotated examples. Data freshness is noted, with a pretraining cutoff of September 2022 and some tuning data up to July 2023.

Finally, the text briefly refers to evaluation results for Llama 1 and Llama 2 on academic benchmarks, utilizing an internal evaluations library. Overall, the information covers various aspects of Llama 2, including model details, development, training, license, intended use, and environmental considerations.

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
    "0": "the meaning of life is to find your gift. you are here to make"
  }
]
```

# **Model Details**

Name: sshleifer/tiny-gpt2
Framework: Transformers (Hugging Face library)
Type: Pretrained language model
Architecture: GPT-2 (Generative Pre-trained Transformer 2)
Description: Brief overview of the model's purpose and capabilities.
Author: sshleifer (the Hugging Face username or contributor)
Repository: Link to the Hugging Face model repository.
Documentation: Link to documentation if available.
Model Characteristics:
Size: Information on the size of the model in terms of parameters.
Training Data: Description of the dataset used for training.
Architecture Details: Information on the architecture, layers, attention mechanisms, etc.
Intended Use and Limitations:
Use Cases: Describe the types of tasks the model is designed for (e.g., text generation, completion, summarization).
Fine-Tuning: Mention if the model is suitable for fine-tuning on specific downstream tasks.
How to Use:
Installation: Instructions for installing and using the model.
Example Code: Code snippets demonstrating how to use the model.
Model Performance:
Evaluation Metrics: If available, mention metrics used to evaluate the model's performance.
Benchmark Results: Any benchmark results or comparisons with other models.
Caveats and Considerations:
Limitations: Note any limitations or known issues with the model.
Special Requirements: Mention any specific requirements or considerations for using the model effectively.

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
    "0": "the meaning of life is confirdit confir reviewing trilogypressoho Daniel antibiotic Daniel Jr confir Prob trilogy intermittent Prob stairs Brew conservation antibiotic Habit substpress stairs ESV autonomyoho intermittentditJD reviewing vendors vendors antibioticRocket antibiotic trilogy trilogy Rh reviewing Prob RhJDpress JrJDdit Probdit ESV"
  }
]
```

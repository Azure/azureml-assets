# **Model Details**

Model Name: EleutherAI/gpt-neo-2.7B

GPT-Neo 2.7B is a transformer model by EleutherAI, replicating the GPT-3 architecture with 2.7 billion parameters. Trained on the Pile dataset, the model underwent 400,000 steps of training on 420 billion tokens, using a masked autoregressive language model and cross-entropy loss.

The intended use of GPT-Neo is text generation from prompts, leveraging its understanding of English for downstream tasks. However, the model's core functionality is predicting the next token in a text string. Limitations include uncertainties in handling prompts, and biases from training on the Pile dataset, known for containing profanity and potentially offensive language. The recommendation is for human curation to filter outputs and enhance quality.

Evaluation results were obtained using EleutherAI's evaluation harness. There are inconsistencies with reported values for GPT-2 and GPT-3, prompting ongoing investigation by EleutherAI. User feedback and further testing of the evaluation harness are encouraged.

In summary, GPT-Neo 2.7B is a powerful language model with notable capabilities, but users are advised to exercise caution due to potential uncertainties in its responses, biases from the training data, and ongoing efforts to address evaluation discrepancies. Human curation is recommended for output filtering to ensure desirable and appropriate results.

Detailed results can be found- https://discord.com/invite/vtRgjbM

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
    "0": "the meaning of life is to be found in the way we live it."
  }
]
```

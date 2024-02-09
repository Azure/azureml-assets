# **Model Details**

Model Name: EleutherAI/gpt-neo-1.3B
GPT-Neo 1.3B is a transformer model created by EleutherAI, replicating the GPT-3 architecture with 1.3 billion parameters. Trained on the Pile dataset, consisting of diverse texts, the model underwent training for 362,000 steps on 380 billion tokens. The training focused on masked autoregressive language modeling using cross-entropy loss.

The intended use of GPT-Neo is for generating text from prompts, leveraging its understanding of the English language to extract features for downstream tasks. Its core functionality is predicting the next token in a given text string. However, the model has limitations, and uncertainties exist in its response to prompts. The training dataset, Pile, contains profanity and potentially offensive language, leading to a warning about the generation of socially unacceptable text. Human curation is recommended for output filtering.

The model's performance is evaluated using various metrics, with average performance reported as 29.44. Task-specific metrics include ARC (25-shot) at 31.23, HellaSwag (10-shot) at 48.47, MMLU (5-shot) at 24.82, TruthfulQA (0-shot) at 39.63, Winogrande (5-shot) at 56.91, GSM8K (5-shot) at 0.45, and DROP (3-shot) at 4.6.

Despite the model's capabilities, users are advised to exercise caution as offensive content may arise without warning. Regular human curation is encouraged to enhance output quality and filter undesirable content.

Detailed results can be found- https://huggingface.co/datasets/open-llm-leaderboard/details_EleutherAI__gpt-neo-1.3B

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
    "0": "the meaning of life is to be found in the life of the soul,"
  }
]
```

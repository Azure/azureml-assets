Falcon-7B is a large language model with 7 billion parameters. It is a causal decoder-only model developed by TII and trained on 1,500 billion tokens of RefinedWeb dataset, which was enhanced with curated corpora. The model is available under the Apache 2.0 license. It outperforms comparable open-source models and features an architecture optimized for inference. However, it is a raw, pretrained model that should be further finetuned for most use cases.

The model is recommended for research on large language models and as a foundation for further specialization and finetuning for specific tasks. It should not be used in production without adequate assessment of risks and mitigation. The model carries biases commonly encountered online and is trained on English and French data only.

The training details of Falcon-7B include information about the training data, training procedure, and hyperparameters used. It was trained on 384 A100 40GB GPUs using a 2D parallelism strategy combined with ZeRO. The model description mentions the architectural adaptations from the GPT-3 model, such as rotary positional embeddings, multiquery attention, and FlashAttention.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/tiiuae/falcon-7b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


### Sample inputs and outputs (for real-time inference)

```json
{
  "input_data": {
      "input_string":["the meaning of life is"]
  }
}
```

#### Sample output
```json
[
  {
    "0": "the meaning of life is to find your gift. the purpose of life is to give it away."
  }
]
```

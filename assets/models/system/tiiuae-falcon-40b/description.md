Falcon-40B is a large language model (LLM) developed by the Technology Innovation Institute (TII) with 40 billion parameters. It is a causal decoder-only model trained on 1 trillion tokens from the RefinedWeb dataset, enhanced with curated corpora. Falcon-40B supports English, German, Spanish, and French languages, with limited capabilities in Italian, Portuguese, Polish, Dutch, Romanian, Czech, and Swedish. It is available under the Apache 2.0 license.

Falcon-40B is considered the best open-source model currently available, optimized for inference with features such as FlashAttention and multiquery. However, it is recommended to fine-tune the model for specific use cases.

The training of Falcon-40B involved using 384 A100 40GB GPUs and took two months. The model carries biases and stereotypes encountered online and requires appropriate precautions for production use. It is suggested to finetune the model for specific tasks and consider guardrails. The technical specifications, training details, and evaluation results are provided in the summary.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/tiiuae/falcon-40b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


### Sample inputs and outputs (for real-time inference)

```json
{
  "input_data": {
      "input_string":["The meaning of the life is"]
  }
}
```

#### Sample output
```json
[
  {
    "0": "The meaning of the life is to find your gift. The purpose of life is to give it away"
  }
]
```

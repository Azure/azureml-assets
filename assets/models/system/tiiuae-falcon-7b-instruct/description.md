# Description
Falcon-7B-Instruct is a causal decoder-only language model with 7 billion parameters. It is based on Falcon-7B and has been fine-tuned on a mixture of instruct and chat datasets. The model is available under the Apache 2.0 license.

# Key Information:

Developed by: TII (The Institute of Imaginary)
Model type: Causal decoder-only
Language(s): English and French
License: Apache 2.0
Finetuned from the base model Falcon-7B

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/tiiuae/falcon-7b-instruct" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.s

# Training Details:

Falcon-7B-Instruct was finetuned on a 250M tokens mixture of instruct/chat datasets.

| **Data source**    | **Fraction** | **Tokens** | **Description**                   |
|--------------------|--------------|------------|-----------------------------------|
| Bai ze             | 65%          | 164B       |   chat                            |
| GPT4All            | 25%          | 62B        |   instruct                        |
| GPTeacher          | 5%           | 11B        |   instruct                        |
| RefinedWeb-English | 5%           | 13B        |  massive web crawl                |

The data was tokenized with the Falcon-7B/40B tokenizer.

## Evaluation
*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.

### Model Architecture:

Falcon-7B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper (Brown et al., 2020), with the following differences:

* **Positionnal embeddings:**  rotary (Su et al., 2021);
* **Attention:** (Shazeer et al., 2019) and FlashAttention (Dao et al., 2022);
* **Decoder-block:**  parallel attention/MLP with a single layer norm.

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 32        |                                        |
| `d_model`          | 4544      | Increased to compensate for multiquery |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

### Hardware 

Falcon-7B-Instruct was trained on AWS SageMaker, on 32 A100 40GB GPUs in P4d instances.

#### Software
Falcon-7B-Instruct was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)

#### License

Falcon-7B-Instruct is available under the Apache 2.0 license.

# Model Evaluation Samples

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>

# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

## Sample input (for real-time inference)

```json
{
  "input_data": {
    "input_string": [
      "Develop a Python function to sort a list of integers in ascending order"
    ],
    "parameters": {
      "top_p": 0.9,
      "temperature": 0.6,
      "do_sample": true,
      "max_new_tokens": 120,
      "return_full_text": false
    }
  }
}
```

## Sample output
```json
[
  {
    "0": "\nYou can use the built-in Python function `sorted()` to sort a list of integers in ascending order. Here is an example:\n\n```python\nmy_list = [10, 8, 5, 2, 9]\nsorted_list = sorted(my_list)\nprint(sorted_list)\n```\nOutput:\n```\n[2, 8, 9, 5, 10]\n```"
  }
]
```
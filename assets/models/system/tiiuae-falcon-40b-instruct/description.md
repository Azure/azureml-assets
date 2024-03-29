# Description
Falcon-40B-Instruct is a large language model with 40 billion parameters, developed by TII. It is a causal decoder-only model fine-tuned on a mixture of Baize data and is released under the Apache 2.0 license. This model is optimized for inference and features FlashAttention and multiquery architectures. It is primarily designed for chat and instruct applications in English and French. However, it may not be suitable for further fine-tuning. It is available under the Apache 2.0 license.

# Key Details:

Model Type: Causal decoder-only
Languages: English and French
License: Apache 2.0
Training Data: Fine-tuned on 150 million tokens from Bai ze mixed with 5% of RefinedWeb data
Architecture: Based on GPT-3 with optimizations including rotary positional embeddings, FlashAttention, and multiquery attention
Hardware: Trained on AWS SageMaker using 64 A100 40GB GPUs in P4d instances
Software: Utilizes a custom distributed training codebase called Gigatron

# Recommendations and Limitations:

Falcon-40B-Instruct may carry biases commonly found online due to its training data.
Users are advised to implement guardrails and take precautions for production use.
It's mostly suited for English and French and may not generalize well to other languages.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/tiiuae/falcon-40b-instruct" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

## Training Details

### Training Data

Falcon-40B-Instruct was finetuned on a 150M tokens from Bai ze mixed with 5% of RefinedWeb data.

The data was tokenized with the Falcon-7B/40B tokenizer.

### Training Procedure 

Falcon-40B-Instruct was trained on AWS SageMaker, on 64 A100 40GB GPUs in P4d instances.


## Evaluation

*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


### Model Architecture and Objective

Falcon-40B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with a two layer norms.

For multiquery, we are using an internal variant which uses independent key and values per tensor parallel degree.

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 60        |                                        |
| `d_model`          | 8192      |                                        |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

#### Hardware

Falcon-40B-Instruct was trained on AWS SageMaker, on 64 A100 40GB GPUs in P4d instances.

#### Software

Falcon-40B-Instruct was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)

#### License

Falcon-40B is made available under the Apache 2.0 license.

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
      "input_string":["Develop a Python function to sort a list of integers in ascending order"]
  }
}
```

## Sample output
```json
[
  {
    "0": "You can use the sorted() function in Python to sort a list of integers in ascending order. Here's an example: my_list = [3,1,6,4,1,5] sorted_list = sorted(my_list) print(sorted_list) This will output: [1,1,3,4,5,6]"
  }
]
```

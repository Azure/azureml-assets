# Description
Falcon-7B-Instruct is a large language model with 7 billion parameters, developed by TII. It is a causal decoder-only model and is released under the Apache 2.0 license. This model is optimized for inference and features FlashAttention and multiquery architectures. It is primarily designed for chat and instruct applications in English and French. However, it may not be suitable for further fine-tuning. It is available under the Apache 2.0 license.

# Key Details:

Model Type: Causal decoder-only
Languages: English and French
License: Apache 2.0
Training Data: Fine-tuned on the Falcon-7B model
Architecture: Based on GPT-3 with optimizations including rotary positional embeddings, FlashAttention, and multiquery attention
Hardware: Falcon-40B was trained on AWS SageMaker, on 32 A100 40GB GPUs in P4d instances.
Software: Utilizes a custom distributed training codebase called Gigatron

# Recommendations and Limitations:

Falcon-7B-Instruct may carry biases commonly found online due to its training data.
Users are advised to implement guardrails and take precautions for production use.
It's mostly suited for English and French and may not generalize well to other languages.

> Review the <a href="https://huggingface.co/tiiuae/falcon-7b-instruct" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

## Training Details

### Training Data

Falcon-40B-Instruct was finetuned on a 164M tokens from Bai ze mixed with 5% of RefinedWeb data.

The data was tokenized with the Falcon-7B/40B tokenizer.

### Training Procedure 

Falcon-40B-Instruct was trained on AWS SageMaker, on 32 A100 40GB GPUs in P4d instances.


## Evaluation

*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


### Model Architecture and Objective

Falcon-7B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with a single layer norm.

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 32        |                                        |
| `d_model`          | 4544      | Increased to compensate for multiquery |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

#### Hardware

Falcon-40B-Instruct was trained on AWS SageMaker, on 32 A100 40GB GPUs in P4d instances.

#### Software

Falcon-40B-Instruct was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)

#### License

Falcon-7B-Instruct is made available under the Apache 2.0 license.


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
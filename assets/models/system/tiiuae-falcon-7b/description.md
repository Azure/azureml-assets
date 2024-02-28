# Description
Falcon-7B, a substantial language model boasting 7 billion parameters, stands out as a causal decoder-only model crafted by TII. Its training involved a vast corpus of 1,500 billion tokens from the RefinedWeb dataset, enriched with carefully curated corpora. Released under the Apache 2.0 license, Falcon-7B surpasses comparable open-source models, offering an architecture fine-tuned for efficient inference. However, it serves as a raw, pretrained model that typically requires additional finetuning to suit diverse use cases.

Primarily recommended for research into large language models, Falcon-7B serves as a foundational resource for further specialization and task-specific finetuning. Caution is advised when considering its deployment in production, necessitating a thorough assessment of risks and the implementation of adequate mitigation measures. Notably, the model inherits biases commonly encountered in online content and is exclusively trained on English and French data.

The training specifics of Falcon-7B encompass comprehensive details regarding the training data, procedure, and hyperparameters. Its training occurred across 384 A100 40GB GPUs, employing a 2D parallelism strategy in conjunction with ZeRO. The model description highlights architectural enhancements from the GPT-3 model, including rotary positional embeddings, multiquery attention, and FlashAttention.


> The summary provided above was created utilizing ChatGPT. To gain insights into the model's training data, evaluation metrics, licensing, intended applications, limitations, and potential biases, it is essential to refer to the <a href="https://huggingface.co/tiiuae/falcon-7b" target="_blank">original model card</a>. Selected content from the model card is presented below for your convenience.

### Training Details

#### Training Data

Falcon-7B was trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), a high-quality filtered and deduplicated web dataset which we enhanced with curated corpora. Significant components from our curated copora were inspired by The Pile ([Gao et al., 2020](https://arxiv.org/abs/2101.00027)).


| **Data source**    | **Fraction** | **Tokens** | **Sources**                       |
|--------------------|--------------|------------|-----------------------------------|
| [RefinedWeb-English](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | 79%          | 1,185B     | massive web crawl                 |
| Books              | 7%           | 110B       |                                   |
| Conversations      | 6%           | 85B        | Reddit, StackOverflow, HackerNews |
| Code               | 3%           | 45B        |                                   |
| RefinedWeb-French  | 3%           | 45B        | massive web crawl                 |
| Technical          | 2%           | 30B        | arXiv, PubMed, USPTO, etc.        |

The data was tokenized with the Falcon-[7B](https://huggingface.co/tiiuae/falcon-7b)/[40B](https://huggingface.co/tiiuae/falcon-40b) tokenizer.

#### Training Procedure 

Falcon-7B was trained on 384 A100 40GB GPUs, using a 2D parallelism strategy (PP=2, DP=192) combined with ZeRO.


| **Hyperparameter** | **Value**  | **Comment**                               |
|--------------------|------------|-------------------------------------------|
| Precision          | `bfloat16` |                                           |
| Optimizer          | AdamW      |                                           |
| Learning rate      | 6e-4       | 4B tokens warm-up, cosine decay to 1.2e-5 |
| Weight decay       | 1e-1       |                                           |
| Z-loss       | 1e-4       |                                           |
| Batch size         | 2304        | 30B tokens ramp-up                         |

#### Speeds, Sizes, Times

Training happened in early March 2023 and took about two weeks.


#### Evaluation

*Paper coming soon*.

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


#### Technical Specifications 

#### Model Architecture and Objective


Falcon-7B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with a single layer norm.


| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 32        |                                        |
| `d_model`          | 4544      | Increased to compensate for multiquery                                       |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |


#### Compute Infrastructure

#### Hardware

Falcon-7B was trained on AWS SageMaker, on 384 A100 40GB GPUs in P4d instances. 

#### Software

Falcon-7B was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)

#### License

Falcon-7B is made available under the Apache 2.0 license.


# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

## Sample input (for real-time inference)

```json
{
  "input_data": [
    "the meaning of life is"
],
"params": {}
}
```

## Sample output
```json
[
    "0":
"string" "the meaning of life is to find your gift. the purpose of life is to give it away."
]
```

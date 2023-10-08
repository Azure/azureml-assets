# Description
Falcon-7B is a large language model with 7 billion parameters. It is a causal decoder-only model developed by TII and trained on 1,500 billion tokens of RefinedWeb dataset, which was enhanced with curated corpora. The model is available under the Apache 2.0 license. It outperforms comparable open-source models and features an architecture optimized for inference. However, it is a raw, pretrained model that should be further finetuned for most use cases.


The model is recommended for research on large language models and as a foundation for further specialization and finetuning for specific tasks. It should not be used in production without adequate assessment of risks and mitigation. The model carries biases commonly encountered online and is trained on English and French data only.

The training details of Falcon-7B include information about the training data, training procedure, and hyperparameters used. It was trained on 384 A100 40GB GPUs using a 2D parallelism strategy combined with ZeRO. The model description mentions the architectural adaptations from the GPT-3 model, such as rotary positional embeddings, multiquery attention, and FlashAttention.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/tiiuae/falcon-7b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model. Some of the content has been made available below.

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


# Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>


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
  "input_data": {
      "input_string":["the meaning of life is"]
  }
}
```

## Sample output
```json
[
  {
    "0": "the meaning of life is to find your gift. the purpose of life is to give it away."
  }
]
```

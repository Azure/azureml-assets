# Model Overview

Arctic is a dense-MoE Hybrid transformer architecture pre-trained from scratch by the Snowflake AI Research Team. We are releasing model checkpoints for both the base and instruct-tuned versions of Arctic under an Apache-2.0 license. This means you can use them freely in your own research, prototypes, and products. Please see our blog [Snowflake Arctic: The Best LLM for Enterprise AI — Efficiently Intelligent, Truly Open](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake) for more information on Arctic and links to other relevant resources such as our series of cookbooks covering topics around training your own custom MoE models, how to produce high-quality training data, and much more.

- **Inputs:** Models input text only.
- **Output:** Models generate text and code only.
- **Model Architecture:** Arctic combines a 10B dense transformer model with a residual 128x3.66B MoE MLP resulting in 480B total and 17B active parameters chosen using a top-2 gating. For more details about Arctic's model Architecture, training process, data, etc. [see our series of cookbooks](https://www.snowflake.com/en/data-cloud/arctic/cookbook/).
- **License:** Apache-2.0.
- **Model developers:** Snowflake AI Research Team.

## Training Data
Snowflake Arctic was pretrained on 3.5 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets.

## Evaluation Results

| Metric                                   | Value  |
|------------------------------------------|--------|
| MMLU                                     | 67.3   |
| GSM8k                                    | 74.2   |
| Spider                                   | 78.9   |
| IFEval                                   | 52.4   |
| Coding - HumanEval+ & MBPP+ -            | 64.3   |

# Inference samples
Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# Sample Inputs and Outputs (for real-time inference)
### Sample Input
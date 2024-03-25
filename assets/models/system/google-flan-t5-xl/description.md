# Description
FLAN-T5 XL is a language model with improved performance and coverage compared to T5. It supports multiple languages and has been fine-tuned on over 1000 additional tasks. It excels in various benchmarks and is intended for research purposes. However, it should be used with caution due to potential risks and biases.

### Model Details:

Model Type: Language model
Supported Languages: A wide range of languages
License: Apache 2.0
Related Models: FLAN-T5 Checkpoints
Original Checkpoints: Original FLAN-T5 Checkpoints
Resources: Research paper, GitHub Repo, Hugging Face FLAN-T5 Docs

### Usage:

Instructions provided for using the model in transformers with PyTorch, CPU, GPU, and different precisions.

### Uses:

Primary use is for research on language models, zero-shot NLP tasks, in-context few-shot learning NLP tasks, advancing fairness and safety research, and understanding limitations of large language models.


## Training Details:

Trained on a mixture of tasks.
Fine-tuned based on pretrained T5.
One fine-tuned Flan model per T5 model size.
Trained on TPU v3 or TPU v4 pods using t5x codebase with JAX.

### Evaluation:

Evaluated on various tasks in multiple languages (1836 in total).
Detailed evaluation results available in the research paper.


### Environmental Impact:

Carbon emissions estimation possible but specific data not provided.
Used Google Cloud TPU Pods (TPU v3 or TPU v4, ≥ 4 chips).


### Citation:

BibTeX citation for the model's research paper is provided.
Please note that FLAN-T5 XL is a powerful language model intended for research but should be used responsibly and with awareness of its potential limitations and ethical considerations.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/google/flan-t5-xl" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model. Some of the content has been made available below.

#### License

google/flan-t5-xl is made available under the Apache 2.0 license.

# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>

# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

## Sample input (for real-time inference)

```json
{"input_data": {"input_string": ["My name is John and I am", "Once upon a time,"], "parameters": {"max_new_tokens":100, "do_sample":true}}}
```

## Sample output
```json
[
  {
    "0": "My name is John and I am a member of the National Alliance on Mental Illness (NAMI) and I want to help those who are in need."
  },
  {
    "0": "Once upon a time, I wrote a lot about my job. And then time passed very fast and I got married and quit my job to become a homemaker. I still write about work and career, but, well, from a different angle now.\n\nA new job was not something I was particularly stoked to hear about in October last year. I had just started feeling like my routine now as a full-time mom to Aryan was beginning to go to some sort of normal. My body was no"
  }
]
```
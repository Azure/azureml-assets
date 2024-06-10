**distilroberta-base** is a distilled version of the [RoBERTa-base model](https://huggingface.co/roberta-base). It follows the same training procedure as [DistilBERT](https://huggingface.co/distilbert-base-uncased).
The code for the distillation process can be found [here](https://github.com/huggingface/transformers/tree/master/examples/distillation). This model is case-sensitive: it makes a difference between english and English.

The model has 6 layers, 768 dimension and 12 heads, totalizing 82M parameters (compared to 125M parameters for RoBERTa-base).
On average DistilRoBERTa is twice as fast as Roberta-base.

# Training Details

DistilRoBERTa was pre-trained on [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/), a reproduction of OpenAI's WebText dataset (it is ~4 times less training data than the teacher RoBERTa). See the [roberta-base model card](https://huggingface.co/roberta-base/blob/main/README.md) for further details on training.

# Evaluation Results

When fine-tuned on downstream tasks, this model achieves the following results (see [GitHub Repo](https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/README.md)):

Glue test results:

| Task | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
|      | 84.0 | 89.4 | 90.8 | 92.5  | 59.3 | 88.3  | 86.6 | 67.9 |


# Limitations and Biases

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). Predictions generated by the model may include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups.

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation you should look at model like GPT2.

The model should not be used to intentionally create hostile or alienating environments for people. The model was not trained to be factual or true representations of people or events, and therefore using the models to generate such content is out-of-scope for the abilities of this model

# Model Evaluation samples

Task | Use case | Dataset | Python sample (Notebook) | CLI with YAML
|--|--|--|--|--|
Fill Mask|Fill Mask|<a href="https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling" target="_blank">rcds/wikipedia-for-mask-filling</a>|<a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/fill-mask-online-endpoint-oss)

# Sample inputs and outputs

### Sample input
```json
{
    "input_data": [
        "Paris is the <mask> of France.",
        "Today is a <mask> day!"
    ]
}
```

### Sample output
```json
[
  "capital",
  "busy"
]
```
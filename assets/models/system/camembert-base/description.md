CamemBERT is a state-of-the-art language model for French based on the RoBERTa model.

It is now available on Hugging Face in 6 different versions with varying number of parameters, amount of pretraining data and pretraining data source domains.

# Training Details

## Training Data
OSCAR or Open Super-large Crawled Aggregated coRpus is a multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the Ungoliant architecture.

## Training Procedure

| Model                                    | #params | Arch. | Training data                     |
| ---------------------------------------- | ------- | ----- | --------------------------------- |
| `camembert-base`                         | 110M    | Base  | OSCAR (138 GB of text)            |
| `camembert/camembert-large`              | 335M    | Large | CCNet (135 GB of text)            |
| `camembert/camembert-base-ccnet`         | 110M    | Base  | CCNet (135 GB of text)            |
| `camembert/camembert-base-wikipedia-4gb` | 110M    | Base  | Wikipedia (4 GB of text)          |
| `camembert/camembert-base-oscar-4gb`     | 110M    | Base  | Subsample of OSCAR (4 GB of text) |
| `camembert/camembert-base-ccnet-4gb`     | 110M    | Base  | Subsample of CCNet (4 GB of text) |

# Evaluation Results


The model developers evaluated CamemBERT using four different downstream tasks for French: part-of-speech (POS) tagging, dependency parsing, named entity recognition (NER) and natural language inference (NLI).


# Limitations and Biases
**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).

This model was pretrained on a subcorpus of OSCAR multilingual corpus. Some of the limitations and risks associated with the OSCAR dataset, which are further detailed in the [OSCAR dataset card](https://huggingface.co/datasets/oscar), include the following: 

> The quality of some OSCAR sub-corpora might be lower than expected, specifically for the lowest-resource languages.

> Constructed from Common Crawl, Personal and sensitive information might be present.

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/fill-mask-online-endpoint-oss)

# Sample inputs and outputs

### Sample input
```json
{
  "input_data": {"input_string": ["Paris is the <mask> of France.", "Today is a <mask> day!"]
    }
}
```

### Sample output
```json
[
  {
    "0": "city"
  },
  {
    "0": "great"
  }
]
```

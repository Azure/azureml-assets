**roberta-large-mnli** is the [RoBERTa large model](https://huggingface.co/roberta-large) fine-tuned on the [Multi-Genre Natural Language Inference (MNLI)](https://huggingface.co/datasets/multi_nli) corpus. The model is a pretrained model on English language text using a masked language modeling (MLM) objective.

This fine-tuned model can be used for zero-shot classification tasks, including zero-shot sentence-pair classification (see the [GitHub repo](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta) for examples) and zero-shot sequence classification.

# Training Details

## Training Data

This model was fine-tuned on the [Multi-Genre Natural Language Inference (MNLI)](https://cims.nyu.edu/~sbowman/multinli/) corpus. Also see the [MNLI data card](https://huggingface.co/datasets/multi_nli) for more information. 

As described in the [RoBERTa large model card](https://huggingface.co/roberta-large): 

> The RoBERTa model was pretrained on the reunion of five datasets:
> 
> - [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038 unpublished books;
> - [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and headers) ;
> - [CC-News](https://commoncrawl.org/2016/10/news-dataset-available/), a dataset containing 63 millions English news articles crawled between September 2016 and February 2019.
> - [OpenWebText](https://github.com/jcpeterson/openwebtext), an opensource recreation of the WebText dataset used to train GPT-2,
> - [Stories](https://arxiv.org/abs/1806.02847), a dataset containing a subset of CommonCrawl data filtered to match the story-like style of Winograd schemas.
>
> Together theses datasets weight 160GB of text.

Also see the [bookcorpus data card](https://huggingface.co/datasets/bookcorpus) and the [wikipedia data card](https://huggingface.co/datasets/wikipedia) for additional information.

## Training Procedure

### Preprocessing

As described in the [RoBERTa large model card](https://huggingface.co/roberta-large): 

> The texts are tokenized using a byte version of Byte-Pair Encoding (BPE) and a vocabulary size of 50,000. The inputs of
> the model take pieces of 512 contiguous token that may span over documents. The beginning of a new document is marked
> with `<s>` and the end of one by `</s>`
> 
> The details of the masking procedure for each sentence are the following:
> - 15% of the tokens are masked.
> - In 80% of the cases, the masked tokens are replaced by `<mask>`.
> - In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
> - In the 10% remaining cases, the masked tokens are left as is.
> 
> Contrary to BERT, the masking is done dynamically during pretraining (e.g., it changes at each epoch and is not fixed).

### Pretraining 

Also as described in the [RoBERTa large model card](https://huggingface.co/roberta-large): 

> The model was trained on 1024 V100 GPUs for 500K steps with a batch size of 8K and a sequence length of 512. The
> optimizer used is Adam with a learning rate of 4e-4, \\(\beta_{1} = 0.9\\), \\(\beta_{2} = 0.98\\) and
> \\(\epsilon = 1e-6\\), a weight decay of 0.01, learning rate warmup for 30,000 steps and linear decay of the learning
> rate after.

# Evaluation Results

The following evaluation information is extracted from the associated [GitHub repo for RoBERTa](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta). 

The model developers report that the model was evaluated on the following tasks and datasets using the listed metrics: 

- **Dataset:** Part of [GLUE (Wang et al., 2019)](https://arxiv.org/pdf/1804.07461.pdf), the General Language Understanding Evaluation benchmark, a collection of 9 datasets for evaluating natural language understanding systems. Specifically, the model was evaluated on the [Multi-Genre Natural Language Inference (MNLI)](https://cims.nyu.edu/~sbowman/multinli/) corpus. See the [GLUE data card](https://huggingface.co/datasets/glue) or [Wang et al. (2019)](https://arxiv.org/pdf/1804.07461.pdf) for further information.
  - **Tasks:** NLI. [Wang et al. (2019)](https://arxiv.org/pdf/1804.07461.pdf) describe the inference task for MNLI as: 
  > The Multi-Genre Natural Language Inference Corpus [(Williams et al., 2018)](https://arxiv.org/abs/1704.05426) is a crowd-sourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. We use the standard test set, for which we obtained private labels from the authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) sections. We also use and recommend the SNLI corpus [(Bowman et al., 2015)](https://arxiv.org/abs/1508.05326) as 550k examples of auxiliary training data.
  - **Metrics:** Accuracy  
  
- **Dataset:** [XNLI (Conneau et al., 2018)](https://arxiv.org/pdf/1809.05053.pdf), the extension of the [Multi-Genre Natural Language Inference (MNLI)](https://cims.nyu.edu/~sbowman/multinli/) corpus to 15 languages: English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu. See the [XNLI data card](https://huggingface.co/datasets/xnli) or [Conneau et al. (2018)](https://arxiv.org/pdf/1809.05053.pdf) for further information.
  - **Tasks:** Translate-test (e.g., the model is used to translate input sentences in other languages to the training language)
  - **Metrics:** Accuracy

**GLUE test results** (dev set, single model, single-task fine-tuning): 90.2 on MNLI

XNLI test results:

| Task | en |  fr | es  | de  | el  | bg  | ru  | tr  | ar  | vi  | th  | zh  | hi  | sw  | ur  |
|:----:|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|      |91.3|82.91|84.27|81.24|81.74|83.13|78.28|76.79|76.64|74.17|74.05| 77.5| 70.9|66.65|66.81|	

# Risks, Limitations and Biases

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). The [RoBERTa large model card](https://huggingface.co/roberta-large) notes that: "The training data used for this model contains a lot of unfiltered content from the internet, which is far from neutral." 

Predictions generated by the model can include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups.

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[text-classification-online-endpoint.ipynb](https://aka.ms/text-classification-online-endpoint-oss)

# Sample inputs and outputs

### Sample input
```json
{
    "input_data": [
        "Today was an amazing day!",
        "It was an unfortunate series of events."
    ]
}
```

### Sample output
```json
[
  {
    "label": "NEUTRAL",
    "score": 0.6557486057281494
  },
  {
    "label": "NEUTRAL",
    "score": 0.7130464911460876
  }
]
```

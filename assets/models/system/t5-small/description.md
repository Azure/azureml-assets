The developers of the Text-To-Text Transfer Transformer (T5) [write](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html): 

> With T5, we propose reframing all NLP tasks into a unified text-to-text-format where the input and output are always text strings, in contrast to BERT-style models that can only output either a class label or a span of the input. Our text-to-text framework allows us to use the same model, loss function, and hyperparameters on any NLP task.

T5-Small is the checkpoint with 60 million parameters. 

# Training Details

## Training Data

The model is pre-trained on the [Colossal Clean Crawled Corpus (C4)](https://www.tensorflow.org/datasets/catalog/c4), which was developed and released in the context of the same [research paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf) as T5.

The model was pre-trained on a on a **multi-task mixture of unsupervised and supervised tasks**.
Thereby, the following datasets were being used for:

#### **Datasets used for Unsupervised denoising objective**:
- [C4](https://huggingface.co/datasets/c4)
- [Wiki-DPR](https://huggingface.co/datasets/wiki_dpr)

#### **Datasets used for Supervised text-to-text language modeling objective**

- Sentence acceptability judgment
  - CoLA [Warstadt et al., 2018](https://arxiv.org/abs/1805.12471)
- Sentiment analysis 
  - SST-2 [Socher et al., 2013](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
- Paraphrasing/sentence similarity
  - MRPC [Dolan and Brockett, 2005](https://aclanthology.org/I05-5002)
  - STS-B [Ceret al., 2017](https://arxiv.org/abs/1708.00055)
  - QQP [Iyer et al., 2017](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- Natural language inference
  - MNLI [Williams et al., 2017](https://arxiv.org/abs/1704.05426)
  - QNLI [Rajpurkar et al.,2016](https://arxiv.org/abs/1606.05250)
  - RTE [Dagan et al., 2005](https://link.springer.com/chapter/10.1007/11736790_9) 
  - CB [De Marneff et al., 2019](https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf)
- Sentence completion
  - COPA [Roemmele et al., 2011](https://www.researchgate.net/publication/221251392_Choice_of_Plausible_Alternatives_An_Evaluation_of_Commonsense_Causal_Reasoning)
- Word sense disambiguation
  - WIC [Pilehvar and Camacho-Collados, 2018](https://arxiv.org/abs/1808.09121)
- Question answering
  - MultiRC [Khashabi et al., 2018](https://aclanthology.org/N18-1023)
  - ReCoRD [Zhang et al., 2018](https://arxiv.org/abs/1810.12885)
  - BoolQ [Clark et al., 2019](https://arxiv.org/abs/1905.10044)

## Training Procedure

In their [abstract](https://jmlr.org/papers/volume21/20-074/20-074.pdf), the model developers write: 

> In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. 

The framework introduced, the T5 framework, involves a training procedure that brings together the approaches studied in the paper. See the [research paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf) for further details.

## Evaluation Results 

For full results for T5-small, see the [research paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf), Table 14.

## Testing Data, Factors & Metrics

The developers evaluated the model on 24 tasks, see the [research paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf) for full details.

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[text-translation-online-endpoint.ipynb](https://aka.ms/translation-online-endpoint-oss)

# Sample inputs and outputs

### Sample input
```json
{
    "input_data": [
        "translate English to French: Life is so beautiful, once you learn how to live with it",
        "translate English to German: Berlin is the capital of Germany"
    ]
}
```

### Sample output
```json
[
  "La vie est tellement belle, une fois que vous en apprendrez comment vivre avec elle",
  "Berlin ist die Hauptstadt Deutschlands"
]
```

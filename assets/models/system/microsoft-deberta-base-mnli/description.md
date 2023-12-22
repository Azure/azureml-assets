[DeBERTa](https://arxiv.org/abs/2006.03654) (Decoding-enhanced BERT with Disentangled Attention) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. It outperforms BERT and RoBERTa on  majority of NLU tasks with 80GB training data. 

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more details and updates.

This model is the base DeBERTa model fine-tuned with MNLI task

# Evaluation Results

We present the dev results on SQuAD 1.1/2.0 and MNLI tasks.

| Model            | SQuAD 1.1 | SQuAD 2.0 | MNLI-m |
| ---------------- | --------- | --------- | ------ |
| RoBERTa-base     | 91.5/84.6 | 83.7/80.5 | 87.6   |
| XLNet-Large      | -/-       | -/80.2    | 86.8   |
| **DeBERTa-base** | 93.1/87.2 | 86.2/83.1 | 88.8   |

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[text-classification-online-endpoint.ipynb](https://aka.ms/text-classification-online-endpoint-oss)

# Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": [
        "Today was an amazing day!",
        "It was an unfortunate series of events."
    ]
}
```

#### Sample output
```json
[
  {
    "label": "NEUTRAL",
    "score": 0.9817705750465393
  },
  {
    "label": "NEUTRAL",
    "score": 0.9873806238174438
  }
]
```

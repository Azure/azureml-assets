[DeBERTa](https://arxiv.org/abs/2006.03654) (Decoding-enhanced BERT with Disentangled Attention) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. It outperforms BERT and RoBERTa on  majority of NLU tasks with 80GB training data. 

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more details and updates.

This is the DeBERTa large model fine-tuned with MNLI task.

# Evaluation Results

We present the dev results on SQuAD 1.1/2.0 and several GLUE benchmark tasks.

| Model                                                                                       | SQuAD 1.1     | SQuAD 2.0     | MNLI-m/mm     | SST-2    | QNLI     | CoLA     | RTE      | MRPC          | QQP           | STS-B         |
| ------------------------------------------------------------------------------------------- | ------------- | ------------- | ------------- | -------- | -------- | -------- | -------- | ------------- | ------------- | ------------- |
|                                                                                             | F1/EM         | F1/EM         | Acc           | Acc      | Acc      | MCC      | Acc      | Acc/F1        | Acc/F1        | P/S           |
| BERT-Large                                                                                  | 90.9/84.1     | 81.8/79.0     | 86.6/-        | 93.2     | 92.3     | 60.6     | 70.4     | 88.0/-        | 91.3/-        | 90.0/-        |
| RoBERTa-Large                                                                               | 94.6/88.9     | 89.4/86.5     | 90.2/-        | 96.4     | 93.9     | 68.0     | 86.6     | 90.9/-        | 92.2/-        | 92.4/-        |
| XLNet-Large                                                                                 | 95.1/89.7     | 90.6/87.9     | 90.8/-        | 97.0     | 94.9     | 69.0     | 85.9     | 90.8/-        | 92.3/-        | 92.5/-        |
| [DeBERTa-Large](https://huggingface.co/microsoft/deberta-large)<sup>1</sup>                 | 95.5/90.1     | 90.7/88.0     | 91.3/91.1     | 96.5     | 95.3     | 69.5     | 91.0     | 92.6/94.6     | 92.3/-        | 92.8/92.5     |
| [DeBERTa-XLarge](https://huggingface.co/microsoft/deberta-xlarge)<sup>1</sup>               | -/-           | -/-           | 91.5/91.2     | 97.0     | -        | -        | 93.1     | 92.1/94.3     | -             | 92.9/92.7     |
| [DeBERTa-V2-XLarge](https://huggingface.co/microsoft/deberta-v2-xlarge)<sup>1</sup>         | 95.8/90.8     | 91.4/88.9     | 91.7/91.6     | **97.5** | 95.8     | 71.1     | **93.9** | 92.0/94.2     | 92.3/89.8     | 92.9/92.9     |
| **[DeBERTa-V2-XXLarge](https://huggingface.co/microsoft/deberta-v2-xxlarge)<sup>1,2</sup>** | **96.1/91.4** | **92.2/89.7** | **91.7/91.9** | 97.2     | **96.0** | **72.0** | 93.5     | **93.1/94.9** | **92.7/90.3** | **93.2/93.1** |
--------

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
    "score": 0.9605958461761475
  },
  {
    "label": "NEUTRAL",
    "score": 0.98270583152771
  }
]
```

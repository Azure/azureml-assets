# Training Details

## Hyperparameters

```
seed=42
batch_size = 12
n_epochs = 4
base_LM_model = "microsoft/MiniLM-L12-H384-uncased"
max_seq_len = 384
learning_rate = 4e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64
grad_acc_steps=4
```

## Evaluation Results
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).
```
"exact": 76.13071675229513,
"f1": 79.49786500219953,
"total": 11873,
"HasAns_exact": 78.35695006747639,
"HasAns_f1": 85.10090269418276,
"HasAns_total": 5928,
"NoAns_exact": 73.91084945332211,
"NoAns_f1": 73.91084945332211,
"NoAns_total": 5945
```

# Model Evaluation samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad_v2" target="_blank">Squad v2</a>|<a href="https://aka.ms/azureml-eval-sdk-question-answering" target="_blank">evaluate-model-question-answering.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-question-answering" target="_blank">evaluate-model-question-answering.yml</a>

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[question-answering-online-endpoint.ipynb](https://aka.ms/question-answering-online-endpoint-oss)

# Sample inputs and outputs

### Sample input
```json
{
    "input_data": {
        "question": "What's my name?",
        "context": "My name is John and I live in Seattle"
    }
}
```

### Sample output
```json
[
  "John"
]
```

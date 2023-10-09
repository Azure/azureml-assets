The "deepset/roberta-base-squad2" model is a version of the RoBERTa model tailored for answering questions using the SQuAD 2.0 dataset. RoBERTa is a strong transformer-based model for natural language processing tasks. This variant has been fine-tuned to excel in answering questions based on provided context from the SQuAD 2.0 dataset. It's designed to provide accurate answers to questions and is useful for question-answering applications.

### Overview
Language model: roberta-base
Language: English
Downstream-task: Extractive QA
Training data: SQuAD 2.0
Eval data: SQuAD 2.0
Code: See an example QA pipeline on Haystack
Infrastructure: 4x Tesla v100

### Hyperparameters
batch_size = 96
n_epochs = 2
base_LM_model = "roberta-base"
max_seq_len = 386
learning_rate = 3e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/deepset/roberta-base-squad2" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-question-answering" target="_blank">question-answering-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-question-answering" target="_blank">question-answering-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-question-answering" target="_blank">question-answering-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>



#### Performance
Evaluated on the SQuAD 2.0 dev set with the official eval script.

"exact": 79.87029394424324,
"f1": 82.91251169582613,

"total": 11873,
"HasAns_exact": 77.93522267206478,
"HasAns_f1": 84.02838248389763,
"HasAns_total": 5928,
"NoAns_exact": 81.79983179142137,
"NoAns_f1": 81.79983179142137,
"NoAns_total": 5945

### Authors
Branden Chan: branden.chan@deepset.ai
Timo Möller: timo.moeller@deepset.ai
Malte Pietsch: malte.pietsch@deepset.ai
Tanay Soni: tanay.soni@deepset.ai
### Model Evaluation

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad_v2" target="_blank">Squad v2</a>|<a href="https://aka.ms/azureml-eval-sdk-question-answering" target="_blank">evaluate-model-question-answering.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-question-answering" target="_blank">evaluate-model-question-answering.yml</a>


#### Sample input
```json
{
    "input_data": {
        "question": ["What is my name?", "Where do I live?"],
        "context": ["My name is John and I live in Seattle.", "My name is Ravi and I live in Hyderabad."]
    }
}
```

#### Sample output
```json
[
    {
        "0": "John"
    },
    {
        "0": "Hyderabad"
    }
]
```

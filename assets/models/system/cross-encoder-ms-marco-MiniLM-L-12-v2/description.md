## Model Description
# Cross-Encoder for MS Marco

This model was trained on the [MS Marco Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) task.

The model can be used for Information Retrieval: Given a query, encode the query will all possible passages (e.g. retrieved with ElasticSearch). Then sort the passages in a decreasing order. See [SBERT.net Retrieve & Re-rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) for more details. The training code is available here: [SBERT.net Training MS Marco](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/ms_marco)


## Usage with Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('model_name')
tokenizer = AutoTokenizer.from_pretrained('model_name')

features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
```


## Usage with SentenceTransformers

The usage becomes easier when you have [SentenceTransformers](https://www.sbert.net/) installed. Then, you can use the pre-trained models like this:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name', max_length=512)
scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])
```


## Performance
In the following table, we provide various pre-trained Cross-Encoders together with their performance on the [TREC Deep Learning 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/) and the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 


| Model-Name        | NDCG@10 (TREC DL 19) | MRR@10 (MS Marco Dev)  | Docs / Sec |
| ------------- |:-------------| -----| --- | 
| **Version 2 models** | | | 
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 69.84 | 32.56 | 9000
| cross-encoder/ms-marco-MiniLM-L-2-v2 | 71.01 | 34.85 | 4100
| cross-encoder/ms-marco-MiniLM-L-4-v2 | 73.04 | 37.70 | 2500
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 74.30 | 39.01 | 1800
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 74.31 | 39.02 | 960
| **Version 1 models** | | | 
| cross-encoder/ms-marco-TinyBERT-L-2  | 67.43 | 30.15  | 9000
| cross-encoder/ms-marco-TinyBERT-L-4  | 68.09 | 34.50  | 2900
| cross-encoder/ms-marco-TinyBERT-L-6 |  69.57 | 36.13  | 680
| cross-encoder/ms-marco-electra-base | 71.99 | 36.41 | 340
| **Other models** | | | 
| nboost/pt-tinybert-msmarco | 63.63 | 28.80 | 2900 
| nboost/pt-bert-base-uncased-msmarco | 70.94 | 34.75 | 340 
| nboost/pt-bert-large-msmarco | 73.36 | 36.48 | 100 
| Capreolus/electra-base-msmarco | 71.23 | 36.89 | 340 
| amberoad/bert-multilingual-passage-reranking-msmarco | 68.40 | 35.54 | 330 
| sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco | 72.82 | 37.88 | 720
 
 Note: Runtime was computed on a V100 GPU.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-fill-mask)|[fill-mask-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-fill-mask)
Batch |[fill-mask-batch-endpoint.ipynb](https://aka.ms/azureml-infer-batch-sdk-fill-mask)|coming soon

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Named Entity Recognition|[Conll2003](https://huggingface.co/datasets/conll2003)|[named-entity-recognition.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[named-entity-recognition.sh](https://aka.ms/azureml-ft-cli-token-classification)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://aka.ms/azureml-ft-cli-extractive-qa)

### Model Evaluation samples

Task | Use case | Dataset | Python sample (Notebook) | CLI with YAML
|--|--|--|--|--|
Fill Mask|Fill Mask|[rcds/wikipedia-for-mask-filling](https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling)|[evaluate-model-fill-mask.ipynb](https://aka.ms/azureml-eval-sdk-fill-mask/)|[evaluate-model-fill-mask.yml](https://aka.ms/azureml-eval-cli-fill-mask/)

### Sample inputs and outputs

#### Sample input
```json
{'input_data': ['I believe the meaning of life is'], 'params': {'top_p': 1.0, 'temperature': 0.8, 'max_new_tokens': 100, 'do_sample': True, 'return_full_text': True}}
```

#### Sample output
```json
["I believe the meaning of life is to give way to you in the present moment to the things you love the most. We don't need to worry about your feelings of guilt, anger, or pain; we need to find ways to make things easier for you and help you get back to normal.\n\nAs a mother, I've always considered that the meaning of the world came from the love we gave each other. I believe that love is a life-sustaining energy that can help us reach our goal of one day"]
```

# Description
The "Cross-Encoder for MS Marco" model is specifically trained for the MS Marco Passage Ranking task, which is a common Information Retrieval task. It is designed to take a query and multiple passages (e.g., retrieved with ElasticSearch) and rank the passages based on their relevance to the query.

# Key Information

Task: MS Marco Passage Ranking
Use Case: Information Retrieval
Model Type: Cross-Encoder
Usage: Given a query and multiple passages, rank the passages by relevance to the query.

# Usage with Transformers (Hugging Face)
This model is particularly useful for ranking passages based on their relevance to a given query, making it suitable for various information retrieval applications.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model. Some of the content has been made available below.

## Performance
In the following table, we provide various pre-trained Cross-Encoders together with their performance on the TREC Deep Learning 2019 and the MS Marco Passage Reranking dataset.

| **Model-Name**         | **NDCG@10 (TREC DL 19)** | **MRR@10 (MS Marco Dev)** | **Docs/Sec** |
|------------------------|--------------------------|---------------------------|--------------| 
| cross-encoder/ms-marco-TinyBERT-L-2-v2| 69.84                    |      32.56 |    9000      |
| cross-encoder/ms-marco-MiniLM-L-2-v2  | 71.01                    |      34.85 |    4100      |
| cross-encoder/ms-marco-MiniLM-L-4-v2  | 73.04                    | 	  37.70 |    2500      |
| cross-encoder/ms-marco-MiniLM-L-6-v2  | 74.30                    |      39.01 |    1800      |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 74.31                    |      39.02 |     960      |
|                                 **Version 1 models**                                         |
| cross-encoder/ms-marco-TinyBERT-L-2	| 67.43                    |      30.15 |    9000      |
| cross-encoder/ms-marco-TinyBERT-L-4	| 68.09                    |      34.50 |    2900      |
| cross-encoder/ms-marco-TinyBERT-L-6	| 69.57                    |      36.13 |     680      |
| cross-encoder/ms-marco-electra-base	| 71.99                    |      36.41 |     340      |
|                                 **Other models**                                             |
| nboost/pt-tinybert-msmarco	        | 63.63                    |      28.80 |     2900     |
| nboost/pt-bert-base-uncased-msmarco	| 70.94                    |      34.75 |      340     |
| nboost/pt-bert-large-msmarco	        | 73.36                    |      36.48 |      100     |
| Capreolus/electra-base-msmarco	    | 71.23                    |      36.89 |      340     |
| amberoad/bert-multilingual-passage-reranking-msmarco|  68.40     |      35.54 |      330     |
|sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco	|72.82 |     37.88  |      720     |


#### License
Falcon-7B is made available under the Apache 2.0 license.


# Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>



# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


## Sample input (for real-time inference)

```json
{
  "input_data": {
      "input_string":["the meaning of life is"]
  }
}
```

## Sample output
```json
[
  {
    "0": "the meaning of life is to find your gift. the purpose of life is to give it away."
  }
]
```
The Cross-Encoder for MS Marco is a model specifically trained on the MS Marco Passage Ranking task. Its primary use case is in Information Retrieval, where given a query, it encodes the query along with all possible passages (retrieved, for example, using ElasticSearch). The encoded passages can then be sorted in decreasing order, aiding in the ranking process. The model's training code is available on SBERT.net, specifically in the training section for MS Marco.

Performance metrics for various pre-trained Cross-Encoders are provided in a table, showcasing their effectiveness on the TREC Deep Learning 2019 and MS Marco Passage Reranking datasets. This information allows users to evaluate and choose the most suitable pre-trained Cross-Encoder for their specific task, such as passage ranking in Information Retrieval. For additional details, users can refer to SBERT.net Retrieve & Re-rank for a more comprehensive understanding of the model's application.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-classification" target="_blank">text-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-classification" target="_blank">text-classification-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-classification" target="_blank">entailment-contradiction-batch.ipynb</a>| coming soon


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text Classification|Textual Entailment|<a href="https://huggingface.co/datasets/glue/viewer/mnli/validation_matched" target="_blank">MNLI</a>|<a href="https://aka.ms/azureml-eval-sdk-text-classification" target="_blank">evaluate-model-text-classification.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-text-classification" target="_blank">evaluate-model-text-classification.yml</a>


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": ["Today was an amazing day!", "It was an unfortunate series of events."]
    }
}
```

#### Sample output
```json
[
  {
    "0": "LABEL_0"
  },
  {
    "0": "LABEL_0"
  }
]
```

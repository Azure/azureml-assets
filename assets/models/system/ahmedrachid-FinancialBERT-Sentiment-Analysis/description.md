FinancialBERT is a BERT-based model pre-trained on a large corpus of financial texts to support financial NLP research and practice in the financial domain. The aim is to provide a resource for financial practitioners and researchers, eliminating the need for substantial computational resources required for model training.

The model underwent fine-tuning for Sentiment Analysis using the Financial PhraseBank dataset. Results from experiments indicate that FinancialBERT surpasses the performance of general BERT and other financial domain-specific models.

Key Details:

Pre-training: FinancialBERT's pre-training process is detailed in a research paper, available at https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining.

Training Data: The model was fine-tuned on the Financial PhraseBank dataset, comprising 4840 financial news articles categorized by sentiment (negative, neutral, positive).

Fine-tuning Hyper-parameters:

Learning rate: 2e-5
Batch size: 32
Max sequence length: 512
Number of training epochs: 5
Evaluation Metrics: The model's performance is assessed using Precision, Recall, and F1-score. A classification report on the test set is available.

FinancialBERT serves as a specialized tool for sentiment analysis in the financial domain, offering improved performance over general BERT and other domain-specific models.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.


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
    "0": "positive"
  },
  {
    "0": "neutral"
  }
]
```

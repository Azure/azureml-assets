fmops/distilbert-prompt-injection model from Hugging Face. this model is based on DistilBERT, which is a smaller and faster version of BERT (Bidirectional Encoder Representations from Transformers), designed for efficient training and deployment. DistilBERT retains much of BERT's performance while using fewer parameters, making it more resource-efficient.

The model predicts that the input text belongs to LABEL_0 with a high confidence score of 99.6%. The second label (LABEL_1) has a very low confidence score of 0.4%.
This suggests that, based on the input text, the model is highly confident in predicting the first label (LABEL_0) and less confident in predicting the second label (LABEL_1)

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/fmops/distilbert-prompt-injection" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-translation" target="_blank">translation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-translation" target="_blank">translation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-translation" target="_blank">translation-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Summarization|News Summary|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">CNN DailyMail</a>|<a href="https://aka.ms/azureml-ft-sdk-news-summary" target="_blank">news-summary.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-news-summary" target="_blank">news-summary.sh</a>
Translation|Translate English to Romanian|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">WMT16</a>|<a href="https://aka.ms/azureml-ft-sdk-translation" target="_blank">translate-english-to-romanian.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-translation" target="_blank">translate-english-to-romanian.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Translation | Translation | <a href="https://huggingface.co/datasets/wmt16/viewer/ro-en/train" target="_blank">wmt16/ro-en</a> | <a href="https://aka.ms/azureml-eval-sdk-translation" target="_blank">evaluate-model-translation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-translation" target="_blank">evaluate-model-translation.yml</a>


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

Summary: camembert-ner is a NER model fine-tuned from camemBERT on the Wikiner-fr dataset and was validated on email/chat data. It shows better performance on entities that do not start with an uppercase. The model has four classes: O, MISC, PER, ORG and LOC. The model can be loaded using HuggingFace. The performance of the model is evaluated using seqeval. Overall, the model has precision 0.8859, recall 0.8971 and f1 0.8914. It shows good performance on PER entities, with precision, recall and f1 of 0.9372, 0.9598 and 0.9483 respectively. The model's author also provided a link to an article on how he used the model results to train a LSTM model for signature detection in emails.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/Jean-Baptiste/camembert-ner" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-token-classification" target="_blank">token-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-token-classification" target="_blank">token-classification-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-token-classification" target="_blank">token-classification-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Token Classification | Token Classification | <a href="https://huggingface.co/datasets/conll2003" target="_blank">CoNLL 2003</a> | <a href="https://aka.ms/azureml-eval-sdk-token-classification" target="_blank">evaluate-model-token-classification.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-token-classification" target="_blank">evaluate-model-token-classification.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": ["Je m'appelle jean-baptiste et je vis à montréal", "george washington est allé à washington"]
    }
}
```

#### Sample output
```json
[
    {
        "0": "['O', 'O', 'I-PER', 'O', 'O', 'O', 'O', 'I-LOC']"
    },
    {
        "0": "['I-PER', 'I-PER', 'O', 'O', 'O', 'I-LOC']"
    }
]
```

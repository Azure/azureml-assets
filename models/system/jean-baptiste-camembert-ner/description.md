Summary: camembert-ner is a NER model fine-tuned from camemBERT on the Wikiner-fr dataset and was validated on email/chat data. It shows better performance on entities that do not start with an uppercase. The model has four classes: O, MISC, PER, ORG and LOC. The model can be loaded using HuggingFace. The performance of the model is evaluated using seqeval. Overall, the model has precision 0.8859, recall 0.8971 and f1 0.8914. It shows good performance on PER entities, with precision, recall and f1 of 0.9372, 0.9598 and 0.9483 respectively. The model's author also provided a link to an article on how he used the model results to train a LSTM model for signature detection in emails.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/Jean-Baptiste/camembert-ner) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[token-classification-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-token-classification)|[token-classification-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-token-classification)
Batch | coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)


### Model Evaluation

| Task                 | Use case             | Dataset                                                 | Python sample (Notebook)                                                                          | CLI with YAML                                                                                   |
|----------------------|----------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Token Classification | Token Classification | [CoNLL 2003](https://huggingface.co/datasets/conll2003) | [evaluate-model-token-classification.ipynb](https://aka.ms/azureml-eval-sdk-token-classification) | [evaluate-model-token-classification.yml](https://aka.ms/azureml-eval-cli-token-classification) |


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["Je m'appelle jean-baptiste et je vis à montréal", "george washington est allé à washington"]
    }
}
```

#### Sample output
```json
[
    [
        {
            "entity_group": "PER",
            "score": 0.99309397,
            "word": "jean-baptiste",
            "start": 12,
            "end": 26
        },
        {
            "entity_group": "LOC",
            "score": 0.99793863,
            "word": "montréal",
            "start": 38,
            "end": 47
        }
    ],
    [
        {
            "entity_group": "PER",
            "score": 0.9873353,
            "word": "george washington",
            "start": 0,
            "end": 17
        },
        {
            "entity_group": "LOC",
            "score": 0.9930083,
            "word": "washington",
            "start": 28,
            "end": 39
        }
    ]
]
```

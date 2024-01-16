The MiniLM-L12-H384-uncased model is a microsoft language model for extractive question answering in English. It was trained on the SQuAD 2.0 dataset and has been evaluated on the SQuAD 2.0 dev set with the official eval script. The model's performance results were an exact match of 76.13 and F1 score of 79.50. The model can be used with Transformers, FARM, or haystack. The model was developed by Vaishali Pal, Branden Chan, Timo Möller, Malte Pietsch, and Tanay Soni, who are employees of deepset, a company focused on bringing NLP to the industry via open source.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/deepset/minilm-uncased-squad2" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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


### Model Evaluation

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad_v2" target="_blank">Squad v2</a>|<a href="https://aka.ms/azureml-eval-sdk-question-answering" target="_blank">evaluate-model-question-answering.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-question-answering" target="_blank">evaluate-model-question-answering.yml</a>


### Sample inputs and outputs (for real-time inference)

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

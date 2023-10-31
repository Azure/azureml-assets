The DistilBERT model is a distilled, smaller, faster, and cheaper version of the BERT model for Transformer-based language model. It is specifically trained for question answering in English and has been fine-tuned using knowledge distillation on SQuAD v1.1. It has 40% less parameters than bert-base-uncased and runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark. However, it is important to note that this model has some limitations and biases and should be used with caution. It should not be used to intentionally create hostile or alienating environments for people and it was not trained to be factual or true representations of people or events. The model has a F1 score of 87.1 on the SQuAD v1.1 dev set, but has unknown environmental impact as the carbon emissions and cloud provider are unknown.  The model was developed by Hugging Face and is licensed under Apache 2.0. The authors of the Model Card are the Hugging Face team.
The DistilBERT model requires significant computational power to train, as it was trained using 8 16GB V100 GPUs and for 90 hours. So it is important to keep that in mind when developing applications that utilize this model.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/distilbert-base-cased-distilled-squad" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Question Answering | Extractive Q&A | <a href="https://huggingface.co/datasets/squad_v2" target="_blank">Squad v2</a> | <a href="https://aka.ms/azureml-eval-sdk-question-answering" target="_blank">evaluate-model-question-answering.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-question-answering" target="_blank">evaluate-model-question-answering.yml</a>


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

The MiniLM-L12-H384-uncased model is a microsoft language model for extractive question answering in English. It was trained on the SQuAD 2.0 dataset and has been evaluated on the SQuAD 2.0 dev set with the official eval script. The model's performance results were an exact match of 76.13 and F1 score of 79.50. The model can be used with Transformers, FARM, or haystack. The model was developed by Vaishali Pal, Branden Chan, Timo Möller, Malte Pietsch, and Tanay Soni, who are employees of deepset, a company focused on bringing NLP to the industry via open source.

The "microsoft/MiniLM-L12-H384-uncased" model is part of the MiniLM model family developed by Microsoft Research. It is designed for natural language processing (NLP) tasks and is particularly known for being lightweight and efficient while still delivering strong performance across various NLP benchmarks.

The "MiniLM-L12-H384-uncased" variant suggests the following:

- "MiniLM": This indicates that the model is based on the MiniLM architecture.
- "L12": This refers to the number of layers in the model. In this case, it has 12 layers.
- "H384": This signifies the hidden size of the model's layers, which is 384.
- "uncased": This indicates that the model doesn't differentiate between uppercase and lowercase letters in its text processing.

Overall, models like MiniLM-L12-H384-uncased are designed to provide a balance between efficiency and performance, making them suitable for various NLP applications where computational resources might be limited. If you're interested in using this model, I recommend checking official Microsoft resources or documentation for more details on its usage and capabilities.

MiniLMv1-L12-H384-uncased: 12-layer, 384-hidden, 12-heads, 33M parameters, 2.7x faster than BERT-Base


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

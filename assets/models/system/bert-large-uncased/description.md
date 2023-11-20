BERT-Large-Uncased is another variant of the BERT (Bidirectional Encoder Representations from Transformers) model, similar to "bert-base-uncased," but with a larger architecture and more parameters. The "uncased" tokenization means that all text is converted to lowercase during preprocessing, just like in the "bert-base-uncased" version.

The main differences between "bert-base-uncased" and "bert-large-uncased" lie in the model size and complexity. BERT-Large-Uncased has more layers and hidden units, making it even more powerful for capturing complex language patterns and semantics. However, this increased complexity also requires more computational resources for training and inference compared to the "bert-base-uncased" version.

BERT-Large-Uncased is well-suited for tasks that demand a deeper understanding of context and a higher level of performance. It excels in a wide range of natural language processing tasks, including text classification, question-answering, sentiment analysis, and more. Researchers and organizations with access to substantial computational resources often prefer using BERT-Large-Uncased to achieve state-of-the-art results in various NLP applications.

<br>Please Note: This model accepts masks in `[mask]` format. See Sample input for reference. 
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/bert-large-uncased" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

BERT-Large, Uncased (Whole Word Masking): 24-layer, 1024-hidden, 16-heads, 340M parameters

|      Model                             | SQUAD 1.1 F1/EM  | Multi NLI Accuracy  |
|----------------------------------------|------------------|---------------------|
|BERT-Large, Uncased (Original)           |   91.0/84.3     |  86.05              |
|BERT-Large, Cased (Whole Word Masking)  |    92.8/86.7     |  87.07              |


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-fill-mask" target="_blank">fill-mask-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-fill-mask" target="_blank">fill-mask-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-fill-mask" target="_blank">fill-mask-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>


### Model Evaluation

Task|Use case|Python sample (Notebook)|CLI with YAML
|--|--|--|--|
Fill Mask|Fill Mask|<a href="https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling" target="_blank">rcds/wikipedia-for-mask-filling</a>|<a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": ["Paris is the [MASK] of France.", "Today is a [MASK] day!"]
    }
}
```

#### Sample output
```json
[
    {
        "0": "capital"
    },
    {
        "0": "good"
    }
]
```

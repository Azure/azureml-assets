Decoding-enhanced BERT with Disentangled Attention is that it is an improvement of the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With 80GB training data, it outperforms the BERT and RoBERTa models in many Natural Language Understanding (NLU) tasks. Key results can be found on the SQuAD 1.1/2.0 and GLUE benchmark tasks when fine-tuned with the MNLI task. The details are available in the official repository and a related paper. If it's useful, cite the paper as described in the citation.
<br>Please Note: This model accepts masks in `[mask]` format. See Sample input for reference. 
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/microsoft/deberta-large" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|coming soon


### Model Evaluation

Task| Use case| Python sample (Notebook)| CLI with YAML
|--|--|--|--|
Fill Mask | Fill Mask | <a href="https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling" target="_blank">rcds/wikipedia-for-mask-filling</a> | <a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


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
        "0": "beautiful"
    }
]
```

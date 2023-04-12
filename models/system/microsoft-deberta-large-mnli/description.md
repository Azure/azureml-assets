DeBERTa is an improvement of BERT and RoBERTa using disentangled attention and enhanced mask decoder. With 80GB training data, it outperforms BERT and RoBERTa on the majority of NLU tasks. The fine-tuned DeBERTa with MNLI task results in the best performance on SQuAD 1.1/2.0 and GLUE benchmark tasks. Further information is available in the official repository and the related paper.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/microsoft/deberta-large-mnli) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[entailment-contradiction-online.ipynb](https://aka.ms/azureml-infer-online-sdk-text-classification)|[text-classification-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-text-classification)
Batch | coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)


### Model Evaluation

| Task                | Use case          | Dataset                                                   | Python sample (Notebook)                                                                        | CLI with YAML                                                                                 |
|---------------------|-------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Text Classification | Emotion Detection | [GoEmotions](https://huggingface.co/datasets/go_emotions) | [evaluate-model-text-classification.ipynb](https://aka.ms/azureml-eval-sdk-text-classification) | [evaluate-model-text-classification.yml](https://aka.ms/azureml-eval-cli-text-classification) |


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["Today was an amazing day!", "It was an unfortunate series of events."]
    }
}
```

#### Sample output
```json
[
    {
        "label": "NEUTRAL",
        "score": 0.9605958461761475
    },
    {
        "label": "NEUTRAL",
        "score": 0.98270583152771
    }
]
```

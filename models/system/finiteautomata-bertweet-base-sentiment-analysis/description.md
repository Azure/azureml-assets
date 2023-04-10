The pysentimiento library is an open-source tool for non-commercial use and scientific research purposes, used for Sentiment Analysis and Social NLP tasks. It was trained on about 40k tweets from the SemEval 2017 corpus, using the BERTweet - a RoBERTa model trained on English tweets and processes POS, NEG, and NEU labels. The Github repository link is https://github.com/finiteautomata/pysentimiento/  While using this model, it is important to keep in mind that they are trained with third-party datasets and are subject to their respective licenses.Additionally Publication details are also mentioned.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[entailment-contradiction-online.ipynb](https://aka.ms/azureml-infer-online-sdk-text-classification)|[text-classification-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-text-classification)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://aka.ms/azureml-ft-cli-extractive-qa)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Text Classification|Emotion Detection|[GoEmotions](https://huggingface.co/datasets/go_emotions)|[evaluate-model-text-classification.ipynb](https://aka.ms/azureml-eval-sdk-text-classification)|


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
        "label": "POS",
        "score": 0.9921929240226746
    },
    {
        "label": "NEG",
        "score": 0.9493512511253357
    }
]
```

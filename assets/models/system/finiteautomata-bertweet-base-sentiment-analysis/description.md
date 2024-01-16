The pysentimiento library is an open-source tool for non-commercial use and scientific research purposes, used for Sentiment Analysis and Social NLP tasks. It was trained on about 40k tweets from the SemEval 2017 corpus, using the BERTweet - a RoBERTa model trained on English tweets and processes POS, NEG, and NEU labels. The Github repository link is https://github.com/finiteautomata/pysentimiento/  While using this model, it is important to keep in mind that they are trained with third-party datasets and are subject to their respective licenses.Additionally Publication details are also mentioned.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-classification" target="_blank">text-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-classification" target="_blank">text-classification-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-classification" target="_blank">entailment-contradiction-batch.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text Classification|Sentiment Classification|<a href="https://huggingface.co/datasets/glue/viewer/sst2/validation" target="_blank">SST2</a>|<a href="https://aka.ms/evaluate-model-sentiment-analysis" target="_blank">evaluate-model-sentiment-analysis.ipynb</a>|<a href="https://aka.ms/evaluate-model-sentiment-analysis-cli" target="_blank">evaluate-model-sentiment-analysis.yml</a>


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
        "0": "POS"
    },
    {
        "0": "NEG"
    }
]
```

Repository: [https://github.com/finiteautomata/pysentimiento/](https://github.com/finiteautomata/pysentimiento/)

Model trained with SemEval 2017 corpus (around ~40k tweets). Base model is [BERTweet](https://github.com/VinAIResearch/BERTweet), a RoBERTa model trained on English tweets.

Uses `POS`, `NEG`, `NEU` labels.

# License

`pysentimiento` is an open-source library for non-commercial use and scientific research purposes only. Please be aware that models are trained with third-party datasets and are subject to their respective licenses.

1. [TASS Dataset license](http://tass.sepln.org/tass_data/download.php)
2. [SEMEval 2017 Dataset license]()

# Model Evaluation samples

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text Classification|Sentiment Classification|<a href="https://huggingface.co/datasets/glue/viewer/sst2/validation" target="_blank">SST2</a>|<a href="https://aka.ms/evaluate-model-sentiment-analysis" target="_blank">evaluate-model-sentiment-analysis.ipynb</a>|<a href="https://aka.ms/evaluate-model-sentiment-analysis-cli" target="_blank">evaluate-model-sentiment-analysis.yml</a>

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[text-classification-online-endpoint.ipynb](https://aka.ms/text-classification-online-endpoint-oss)

# Sample inputs and outputs

### Sample input
```json
{
    "input_data": [
        "Today was an amazing day!",
        "It was an unfortunate series of events."
    ]
}
```

### Sample output
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

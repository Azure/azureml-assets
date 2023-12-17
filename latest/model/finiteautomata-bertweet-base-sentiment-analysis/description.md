Repository: [https://github.com/finiteautomata/pysentimiento/](https://github.com/finiteautomata/pysentimiento/)

Model trained with SemEval 2017 corpus (around ~40k tweets). Base model is [BERTweet](https://github.com/VinAIResearch/BERTweet), a RoBERTa model trained on English tweets.

Uses `POS`, `NEG`, `NEU` labels.

# License

`pysentimiento` is an open-source library for non-commercial use and scientific research purposes only. Please be aware that models are trained with third-party datasets and are subject to their respective licenses.

1. [TASS Dataset license](http://tass.sepln.org/tass_data/download.php)
2. [SEMEval 2017 Dataset license]()

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

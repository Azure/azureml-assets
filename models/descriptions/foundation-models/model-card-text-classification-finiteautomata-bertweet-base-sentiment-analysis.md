The pysentimiento library is an open-source tool for non-commercial use and scientific research purposes, used for Sentiment Analysis and Social NLP tasks. It was trained on about 40k tweets from the SemEval 2017 corpus, using the BERTweet - a RoBERTa model trained on English tweets and processes POS, NEG, and NEU labels. The Github repository link is https://github.com/finiteautomata/pysentimiento/  While using this model, it is important to keep in mind that they are trained with third-party datasets and are subject to their respective licenses.Additionally Publication details are also mentioned.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Text Classification||[GoEmotions](https://huggingface.co/datasets/go_emotions)|[evaluate-model-text-classification.ipynb](https://aka.ms/azureml-eval-sdk-text-classification)|


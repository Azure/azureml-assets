T5 Small is a text-to-text transformer model with 60 million parameters. It is developed by a group of researchers and is based on the Text-To-Text Transfer Transformer (T5) framework, which allows for a unified text-to-text format for input and output of all NLP tasks. T5-Small can be trained for multiple NLP tasks such as machine translation, document summarization, question answering, classification, and even regression tasks. It is pre-trained on the Colossal Clean Crawled Corpus (C4) and a multi-task mixture of unsupervised and supervised tasks in various languages, such as English, French, Romanian and German. You can use the code provided to get started with the model and refer to the resources provided for more information on the model.

> The above summary was generated using ChatGPT. Review the [original-model-card](https://huggingface.co/t5-small) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Summarization|Summarization|[CNN DailyMail](https://huggingface.co/datasets/cnn_dailymail)|[news-summary.ipynb](https://aka.ms/azureml-ft-sdk-news-summary)|[news-summary.sh](https://aka.ms/azureml-ft-cli-news-summary)
Translation|Translation|[WMT16](https://huggingface.co/datasets/cnn_dailymail)|[translation.ipynb](https://aka.ms/azureml-ft-sdk-translation)|[translation.sh](https://aka.ms/azureml-ft-cli-translation)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Translation||[wmt19/cs-en](https://huggingface.co/datasets/wmt19/viewer/cs-en/)|[evaluate-model-translation.ipynb](https://aka.ms/azureml-eval-sdk-translation)|








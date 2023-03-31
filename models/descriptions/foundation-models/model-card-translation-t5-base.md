T5 Base is a text-to-text transformer model that can be used for a variety of NLP tasks, such as machine translation, document summarization, question answering and classification tasks, such as sentiment analysis. It was developed by a team at Google and is pre-trained on the Colossal Clean Crawled Corpus. It is licensed under Apache 2.0 and you can start using the model by installing the T5 tokenizer and model and following the examples provided in the Colab Notebook created by its developers. Be mindful of bias, risks and limitations that may arise while using this model.

> The above summary was generated using ChatGPT. Review the [original-model-card](https://huggingface.co/t5-base) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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






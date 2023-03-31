The RoBERTa Large model is a large transformer-based language model that was developed by the Hugging Face team. It is pre-trained on masked language modeling and can be used for tasks such as sequence classification, token classification, or question answering. Its primary usage is as a fine-tuning tool and is case-sensitive. Additionally, there are metrics provided for DistilBART models, including the number of parameters, inference time, speedup, Rouge 2, and Rouge-L. The distilbart-xsum-12-6 model is recommended with 306 million parameters, 137 milliseconds inference time, 1.68 speedup, 22.12 Rouge 2, and 36.99 Rouge-L.

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
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://aka.ms/azureml-ft-cli-extractive-qa)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Summarization||[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail)|[evaluate-model-summarization.ipynb](https://aka.ms/azureml-eval-sdk-summarization)|


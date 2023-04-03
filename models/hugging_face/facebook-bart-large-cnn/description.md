The BART model is a transformer encoder-encoder model trained on English language data, and fine-tuned on CNN Daily Mail. It is used for text summarization and has been trained to reconstruct text that has been corrupted using an arbitrary noising function. The model is effective for text generation tasks such as summarization, and works well for comprehension tasks such as text classification and question answering. It can be used with the pipeline API in Python, as detailed in the code snippet provided. Lastly, it was introduced in the paper "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" by Lewis et al.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/facebook/bart-large-cnn) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[summarization-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-summarization)|[summarization-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-summarization)
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
|Summarization|Summarization|[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail)|[evaluate-model-summarization.ipynb](https://aka.ms/azureml-eval-sdk-summarization)|

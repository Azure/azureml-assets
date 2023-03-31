The T5-Large is a text-to-text transfer transformer (T5) model with 770 million parameters. It has been developed by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. The T5 model is a language model that is pre-trained on a multi-task mixture of unsupervised and supervised tasks. The model is versatile and can be used for many NLP tasks such as translation, summarization, question answering, and classification. The license of T5-Large is Apache 2.0. The model is made available on GitHub and is well-documented on Hugging Face.  A code sample is provided to get started with this model. The input and output are always text strings. The T5 framework was introduced to bring together transfer learning techniques for NLP and convert all language problems into the text to text format. The T5 model was pre-trained on the Colossal Clean Crawled Corpus (C4). The full details of the training procedure can be found in the research paper. The evaluation of T5-Large is not provided. The information regarding bias, risks, limitations, and recommendations is not available. The authors of the Model Card for T5-Large are not specified.

> The above summary was generated using ChatGPT. Review the [original-model-card](https://huggingface.co/t5-large) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Summarization|Summarization|[CNN DailyMail](https://huggingface.co/datasets/cnn_dailymail)|[news-summary.ipynb](https://github.com/Azure/azureml-examples/tree/sitaram/finetunenotebooks/sdk/python/foundation-models/system/finetune/summarization/news-summary.ipynb)|[news-summary.sh](https://github.com/Azure/azureml-examples/blob/sitaram/finetunenotebooks/cli/foundation-models/system/finetune/summarization/news-summary.sh)
Translation|Translation|[WMT16](https://huggingface.co/datasets/cnn_dailymail)|[translation.ipynb](https://github.com/Azure/azureml-examples/tree/sitaram/finetunenotebooks/sdk/python/foundation-models/system/finetune/translation/translation.ipynb)|[translation.sh](https://github.com/Azure/azureml-examples/blob/sitaram/finetunenotebooks/cli/foundation-models/system/finetune/translation/translation.sh)








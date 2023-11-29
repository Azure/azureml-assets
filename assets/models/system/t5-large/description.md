The T5-Large is a text-to-text transfer transformer (T5) model with 770 million parameters. It has been developed by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. The T5 model is a language model that is pre-trained on a multi-task mixture of unsupervised and supervised tasks. The model is versatile and can be used for many NLP tasks such as translation, summarization, question answering, and classification. The license of T5-Large is Apache 2.0. The model is made available on GitHub and is well-documented on Hugging Face.  A code sample is provided to get started with this model. The input and output are always text strings. The T5 framework was introduced to bring together transfer learning techniques for NLP and convert all language problems into the text to text format. The T5 model was pre-trained on the Colossal Clean Crawled Corpus (C4). The full details of the training procedure can be found in the research paper. The evaluation of T5-Large is not provided. The information regarding bias, risks, limitations, and recommendations is not available. The authors of the Model Card for T5-Large are not specified.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/t5-large" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-translation" target="_blank">translation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-translation" target="_blank">translation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-translation" target="_blank">translation-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Summarization|News Summary|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">CNN DailyMail</a>|<a href="https://aka.ms/azureml-ft-sdk-news-summary" target="_blank">news-summary.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-news-summary" target="_blank">news-summary.sh</a>
Translation|Translate English to Romanian|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">WMT16</a>|<a href="https://aka.ms/azureml-ft-sdk-translation" target="_blank">translate-english-to-romanian.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-translation" target="_blank">translate-english-to-romanian.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Translation | Translation | <a href="https://huggingface.co/datasets/wmt16/viewer/ro-en/train" target="_blank">wmt16/ro-en</a> | <a href="https://aka.ms/azureml-eval-sdk-translation" target="_blank">evaluate-model-translation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-translation" target="_blank">evaluate-model-translation.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["My name is John and I live in Seattle", "Berlin is the capital of Germany."]
    },
    "parameters": {
        "task_type": "translation_en_to_fr"
    }
}
```

#### Sample output
```json
[
    {
        "0": "Mein Name ist John und ich lebe in Seattle."
    },
    {
        "0": "Berlin ist die Hauptstadt Deutschlands."
    }
]
```

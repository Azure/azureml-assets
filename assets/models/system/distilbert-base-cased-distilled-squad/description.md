DistilBERT, a remarkable small-scale Transformer model, emerges as a swift, cost-effective, and lightweight rendition of BERT. It is achieved through the distillation process from BERT base. Impressively, DistilBERT wields a 40% reduction in parameters compared to bert-base-uncased, resulting in a 60% acceleration in processing speed, all the while retaining over 95% of BERT's performance measured against the GLUE language understanding benchmark.

This specific model, "distilbert-base-cased-distilled-squad," is fine-tuned for question answering. Developed by Hugging Face, it falls under the category of Transformer-based language models, and it caters primarily to English.

Accessing this model is facilitated by the Hugging Face Transformers library. It excels in question answering tasks and is demonstrated through Python code samples.

A few critical points deserve attention: the model should not be wielded to generate hostile or harmful content, and its application is restricted to its established abilities. Users should be cognizant of potential biases, risks, and limitations.

The training of the distilbert-base-cased model leverages similar data as its uncased counterpart, sourced from BookCorpus and English Wikipedia. Its effectiveness is validated through an F1 score of 87.1 on the SQuAD v1.1 dev set, comparable to BERT's performance.

Concerning environmental impact, training specifics indicate the use of 8 16GB V100 GPUs over 90 hours. However, details about carbon emissions and cloud providers remain undisclosed.

DistilBERT stands as a testament to advancements in NLP, presenting a lean yet formidable solution that resonates with efficiency and competence.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/distilbert-base-cased-distilled-squad" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-question-answering" target="_blank">question-answering-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-question-answering" target="_blank">question-answering-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-question-answering" target="_blank">question-answering-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Question Answering | Extractive Q&A | <a href="https://huggingface.co/datasets/squad_v2" target="_blank">Squad v2</a> | <a href="https://aka.ms/azureml-eval-sdk-question-answering" target="_blank">evaluate-model-question-answering.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-question-answering" target="_blank">evaluate-model-question-answering.yml</a>


#### Sample input
```json
{
    "inputs": {
        "question": ["What is my name?", "Where do I live?"],
        "context": ["My name is John and I live in Seattle.", "My name is Ravi and I live in Hyderabad."]
    }
}
```

#### Sample output
```json
[
    {
        "0": "John"
    },
    {
        "0": "Hyderabad"
    }
]
```

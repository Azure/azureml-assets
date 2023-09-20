RoBERTa is a transformer-based language model that was fine-tuned from RoBERTa large model on Multi-Genre Natural Language Inference (MNLI) corpus for English. It can be used for zero-shot classification tasks and can be accessed from GitHub Repo. It is important to note that the model was trained on unfiltered data, so generated results may have disturbing and offensive stereotypes. Also, it should not be used to create hostile or alienating environments or to present factual or true representation of people or events.  The model is intended to be used on tasks that use the whole sentence or sequence classification, token classification or question answering. While it is pre-trained on a large corpus of English data, it's only intended to be fine-tuned further on specific tasks. The data the model was trained on includes 160GB of English text from various sources including Wikipedia and news articles. Pretraining was done using V100 GPUs for 500,000 steps using the masked language modeling objective. The training procedure has also been mentioned that it was done using Adam optimizer with a batch size of 8,000 and a sequence length of 512. Additionally, the model is a case-sensitive model, the tokenization and masking is done for the model pre-processing. A pipeline for masked language modeling can be used to directly use the model, but it is highly recommended to use the fine-tuned models available on the model hub for the task that interests you. The bias from the training data will also affect all fine-tuned versions of this model. 
<br>Please Note: This model accepts masks in `<mask>` format. See Sample input for reference. 
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/roberta-base" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-fill-mask" target="_blank">fill-mask-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-fill-mask" target="_blank">fill-mask-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-fill-mask" target="_blank">fill-mask-batch-endpoint.ipynb</a>| coming soon


### Model Evaluation

Task | Use case | Dataset Python sample (Notebook) |CLI with YAML
|--|--|--|--|
Fill Mask | Fill Mask | <a href="https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling" target="_blank">rcds/wikipedia-for-mask-filling</a> | <a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": ["Paris is the <mask> of France.", "Today is a <mask> day!"]
    }
}
```

#### Sample output
```json
[
    {
        "0": "capital"
    },
    {
        "0": "beautiful"
    }
]
```

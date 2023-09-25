The RoBERTa Large model is a pretrained language model developed by the Hugging Face team, based on the transformer architecture. It was trained on a large corpus of English data in a self-supervised manner using the masked language modeling (MLM) objective. The model is case-sensitive and primarily intended for use in fine-tuning downstream tasks such as sequence classification, token classification, or question answering. It was trained on a combination of five datasets weighing 160GB of text, and uses a vocabulary size of 50,000 for tokenization. The model was trained for 500K steps on 1024 V100 GPUs with a batch size of 8K and a sequence length of 512. The optimizer used was Adam with a learning rate of 4e-4, β1=0.9, β2=0.98, and ϵ=1e-6, with a weight decay of 0.01 and learning rate warmup for 30,000 steps.
<br>Please Note: This model accepts masks in `<mask>` format. See Sample input for reference. 
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/roberta-large" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-fill-mask" target="_blank">fill-mask-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-fill-mask" target="_blank">fill-mask-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-fill-mask" target="_blank">fill-mask-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>


### Model Evaluation

Task| Use case| Python sample (Notebook)| CLI with YAML
|--|--|--|--|
Fill Mask | Fill Mask | <a href="https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling" target="_blank">rcds/wikipedia-for-mask-filling</a> | <a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


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

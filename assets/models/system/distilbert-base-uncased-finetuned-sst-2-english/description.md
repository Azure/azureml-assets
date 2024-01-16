This is a fine-tuned version of DistilBERT-base-uncased, trained on SST-2, which reached 91.3 % accuracy on the dev set. Developed by Hugging Face, it's mainly intended to be used for topic classification and can be fine-tuned on downstream tasks, but it's important to keep in mind that it has certain biases, such as biased predictions for certain underrepresented populations and that it should not be used to create hostile or alienating environments for people. Additionally, the authors used the Stanford Sentiment Treebank(sst2) corpora for training the model. 
It's recommended to evaluate the risks of this model by thoroughly probing the bias evaluation datasets like WinoBias, WinoGender, Stereoset


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-classification" target="_blank">text-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-classification" target="_blank">text-classification-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-classification" target="_blank">entailment-contradiction-batch.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://aka.ms/azureml-ft-sdk-emotion-detection" target="_blank">emotion-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-emotion-detection" target="_blank">emotion-detection.sh</a>
Token Classification|Named Entity Recognition|<a href="https://huggingface.co/datasets/conll2003" target="_blank">Conll2003</a>|<a href="https://aka.ms/azureml-ft-sdk-token-classification" target="_blank">named-entity-recognition.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-token-classification" target="_blank">named-entity-recognition.sh</a>
Question Answering|Extractive Q&A|<a href="https://huggingface.co/datasets/squad" target="_blank">SQUAD (Wikipedia)</a>|<a href="https://aka.ms/azureml-ft-sdk-extractive-qa" target="_blank">extractive-qa.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-extractive-qa" target="_blank">extractive-qa.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text Classification|Sentiment Classification|<a href="https://huggingface.co/datasets/glue/viewer/sst2/validation" target="_blank">SST2</a>|<a href="https://aka.ms/evaluate-model-sentiment-analysis" target="_blank">evaluate-model-sentiment-analysis.ipynb</a>|<a href="https://aka.ms/evaluate-model-sentiment-analysis-cli" target="_blank">evaluate-model-sentiment-analysis.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": ["Today was an amazing day!", "It was an unfortunate series of events."]
    }
}
```

#### Sample output
```json
[
    {
        "0": "POSITIVE"
    },
    {
        "0": "NEGATIVE"
    }
]
```

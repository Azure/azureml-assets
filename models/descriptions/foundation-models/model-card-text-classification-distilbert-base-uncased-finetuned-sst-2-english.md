This is a fine-tuned version of DistilBERT-base-uncased, trained on SST-2, which reached 91.3 % accuracy on the dev set. Developed by Hugging Face, it's mainly intended to be used for topic classification and can be fine-tuned on downstream tasks, but it's important to keep in mind that it has certain biases, such as biased predictions for certain underrepresented populations and that it should not be used to create hostile or alienating environments for people. Additionally, the authors used the Stanford Sentiment Treebank(sst2) corpora for training the model. 
It's recommended to evaluate the risks of this model by thoroughly probing the bias evaluation datasets like WinoBias, WinoGender, Stereoset

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://github.com/Azure/azureml-examples/tree/sitaram/finetunenotebooks/sdk/python/foundation-models/system/finetune/token-classification/token-classification.ipynb)|[token-classification.sh](https://github.com/Azure/azureml-examples/blob/sitaram/finetunenotebooks/cli/foundation-models/system/finetune/token-classification/token-classification.sh)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://github.com/Azure/azureml-examples/blob/sitaram/finetunenotebooks/cli/foundation-models/system/finetune/question-answering/extractive-qa.sh)

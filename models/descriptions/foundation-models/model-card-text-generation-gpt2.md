GPT-2 is a transformer-based language model intended for AI researchers and practitioners. It was trained on unfiltered content from Reddit and may have biases. It is best used for text generation, but the training data has not been publicly released. It has several limitations and should be used with caution in situations that require truth and in systems that interact with humans. There are different versions of the model, including GPT-Large, GPT-Medium, and GPT-Xl, available for different use cases. The information provided by the OpenAI team is to complete and give specific examples of bias in their model.

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://github.com/Azure/azureml-examples/tree/sitaram/finetunenotebooks/sdk/python/foundation-models/system/finetune/token-classification/token-classification.ipynb)|[token-classification.sh](https://github.com/Azure/azureml-examples/blob/sitaram/finetunenotebooks/cli/foundation-models/system/finetune/token-classification/token-classification.sh)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Text generation||todo|[evaluate-model-text-generation.ipynb](https://aka.ms/azureml-eval-sdk-text-generation/)|
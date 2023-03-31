GPT-2 is a transformer-based language model intended for AI researchers and practitioners. It was trained on unfiltered content from Reddit and may have biases. It is best used for text generation, but the training data has not been publicly released. It has several limitations and should be used with caution in situations that require truth and in systems that interact with humans. There are different versions of the model, including GPT-Large, GPT-Medium, and GPT-Xl, available for different use cases. The information provided by the OpenAI team is to complete and give specific examples of bias in their model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Text generation||todo|[evaluate-model-text-generation.ipynb](https://aka.ms/azureml-eval-sdk-text-generation/)|

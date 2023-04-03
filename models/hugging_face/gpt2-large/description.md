The OpenAI GPT-2 is a language model that is intended to be used primarily by AI researchers and practitioners. It is capable of performing various uses, including writing assistance and creative writing, but is not recommended to be deployed in human interaction systems without a thorough study of its biases. The training data used to create this model was scraped from Reddit, excluding all pages of Wikipedia, and has not been publicly released. The model was trained on a very large corpus of English data in a self-supervised fashion, meaning it was pretrained on raw texts without human labeling. The evaluation information for this model comes from its associated paper and is evaluated on various language model benchmarks. The results are reported using invertible de-tokenizers to remove pre-processing artifacts.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/gpt2-large) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

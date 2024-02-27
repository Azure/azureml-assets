DistilBERT, a transformers model, is designed to be smaller and quicker than BERT. It underwent pretraining on the same dataset in a self-supervised manner, utilizing the BERT base model as a reference. This entails training solely on raw texts, without human annotation, thus enabling the utilization of vast amounts of publicly accessible data. An automated process generates inputs and labels from these texts, guided by the BERT base model. Specifically, the pretraining process involved three objectives:

Distillation loss: The model was trained to produce probabilities akin to those of the BERT base model.
Masked language modeling (MLM): This constitutes a segment of the original training loss in the BERT base model. By randomly masking 15% of the words in a sentence, the model processes the entire masked sentence and endeavors to predict the masked words. This methodology differs from traditional recurrent neural networks (RNNs) or autoregressive models like GPT, which handle words sequentially or internally mask future tokens. MLM facilitates the acquisition of a bidirectional sentence representation by the model.
Cosine embedding loss: The model was also trained to generate hidden states that closely resemble those of the BERT base model.
In this manner, the model acquires a comparable internal representation of the English language to that of its teacher model, while being more efficient for inference or subsequent tasks.
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/distilbert-base-cased" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

  "input_data": [ 

    "Paris is [MASK] of France" 

  ], 

  "params": {} 

} 
```

#### Sample output
```json
[
  "part"
]
```

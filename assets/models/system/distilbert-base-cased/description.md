DistilBERT is a transformers-based model that offers a more compact and faster alternative to BERT. It underwent self-supervised pretraining using the BERT base model as a reference. This pretraining involved raw text data without human annotations, allowing it to utilize publicly available data. The training process incorporated three main objectives:

1. Distillation Loss: The model was trained to produce probabilities consistent with the BERT base model.
2. Masked Language Modeling (MLM): This loss function, present in the original BERT training, involves randomly masking 15% of words in a sentence, then predicting those masked words. It enables the model to understand sentence context bidirectionally.
3. Cosine Embedding Loss: The model learned to generate hidden states closely resembling those of the BERT base model.

Through these objectives, the model acquires the same core understanding of the English language as its teacher model (BERT), while being more efficient for tasks requiring quick inference or downstream processing.

Intended for uses such as masked language modeling or next sentence prediction, DistilBERT is primarily designed to be fine-tuned for specific tasks. The model hub offers fine-tuned versions tailored to various tasks. Notably, DistilBERT is best suited for tasks that utilize entire sentences (including potentially masked parts) for decision-making, such as sequence classification, token classification, or question answering. For text generation tasks, other models like GPT-2 are more appropriate.



"distilbert-base-cased" refers to a specific variant of the DistilBERT model, which is a compact and efficient version of the BERT (Bidirectional Encoder Representations from Transformers) model. DistilBERT is designed to provide similar performance to BERT while using fewer computational resources, making it well-suited for various natural language processing (NLP) tasks.

Breaking down the name "distilbert-base-cased":

- "distilbert": This indicates that the model is based on the DistilBERT architecture.
- "base": This suggests that it's a base variant, which may have fewer layers and parameters compared to larger variants.
- "cased": This indicates that the model retains the case information of the input text, differentiating between uppercase and lowercase letters.

The DistilBERT model is a smaller, faster version of the BERT model for Transformer-based language modeling with 40% fewer parameters and 60% faster run time while retaining 95% of BERT's performance on the GLUE language understanding benchmark. This English language question answering model has a F1 score of 87.1 on SQuAD v1.1 and was developed by Hugging Face under the Apache 2.0 license. Training the model requires significant computational power, such as 8 16GB V100 GPUs and 90 hours. Intended uses include fine-tuning on downstream tasks, but it should not be used to create hostile or alienating environments and limitations and biases should be taken into account.

 
<br>Please Note: This model accepts masks in `[mask]` format. See Sample input for reference. 
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
    "inputs": {
        "input_string": ["Paris is the [MASK] of France.", "Today is a [MASK] day!"]
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

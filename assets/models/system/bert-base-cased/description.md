BERT, the Bidirectional Encoder Representations from Transformers, is a sophisticated language model pretrained on a vast corpus of English data using a self-supervised approach. It automatically generates inputs and labels from raw texts without human annotations. During pretraining, BERT achieves two main objectives: Masked Language Modeling (MLM) by predicting masked words in a sentence, enabling bidirectional understanding, and Next Sentence Prediction (NSP) to grasp sentence relationships. This process forms a deep internal representation of English, providing valuable features for downstream tasks like text classification and sentiment analysis, making it a powerful tool for developing robust natural language processing systems. This model is intended to be used as a tool for fine-tuning in various downstream tasks such as sequence classification and question answering, but is not recommended for tasks such as text generation. To use this model, you can utilize a pipeline specifically designed for masked language modeling.

<br>Please Note: This model accepts masks in `[mask]` format. See Sample input for reference. 
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/bert-base-cased" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.
### Training Procedure

Preprocessing
The texts are tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are then of the form:

[CLS] Sentence A [SEP] Sentence B [SEP]

With probability 0.5, sentence A and sentence B correspond to two consecutive sentences in the original corpus and in the other cases, it's another random sentence in the corpus. Note that what is considered a sentence here is a consecutive span of text usually longer than a single sentence. The only constrain is that the result with the two "sentences" has a combined length of less than 512 tokens.

The details of the masking procedure for each sentence are the following:

15% of the tokens are masked.
In 80% of the cases, the masked tokens are replaced by [MASK].
In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
In the 10% remaining cases, the masked tokens are left as is.

### Pretraining
The model was trained on 4 cloud TPUs in Pod configuration (16 TPU chips total) for one million steps with a batch size of 256. The sequence length was limited to 128 tokens for 90% of the steps and 512 for the remaining 10%. The optimizer used is Adam with a learning rate of 1e-4, Beta1=0.9,Beta2=0.999, a weight decay of 0.01, learning rate warmup for 10,000 steps and linear decay of the learning rate after.


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

Task | Use case | Dataset | Python sample (Notebook) | CLI with YAML
|--|--|--|--|--|
Fill Mask|Fill Mask|<a href="https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling" target="_blank">rcds/wikipedia-for-mask-filling</a>|<a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
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

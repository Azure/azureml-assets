DistilBERT, a transformers model, is designed to be smaller and quicker than BERT. It underwent pretraining on the same dataset in a self-supervised manner, utilizing the BERT base model as a reference. This entails training solely on raw texts, without human annotation, thus enabling the utilization of vast amounts of publicly accessible data. An automated process generates inputs and labels from these texts, guided by the BERT base model. Specifically, the pretraining process involved three objectives:

Distillation loss: The model was trained to produce probabilities akin to those of the BERT base model.
Masked language modeling (MLM): This constitutes a segment of the original training loss in the BERT base model. By randomly masking 15% of the words in a sentence, the model processes the entire masked sentence and endeavors to predict the masked words. This methodology differs from traditional recurrent neural networks (RNNs) or autoregressive models like GPT, which handle words sequentially or internally mask future tokens. MLM facilitates the acquisition of a bidirectional sentence representation by the model.
Cosine embedding loss: The model was also trained to generate hidden states that closely resemble those of the BERT base model.
In this manner, the model acquires a comparable internal representation of the English language to that of its teacher model, while being more efficient for inference or subsequent tasks.

# Training Details

## Training data

DistilBERT was pretrained on the same data as BERT, which includes the BookCorpus dataset (consisting of 11,038 unpublished books) and English Wikipedia (excluding lists, tables, and headers).

## Training Procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece with a vocabulary size of 30,000.
The model inputs are structured as follows: [CLS] Sentence A [SEP] Sentence B [SEP]
With a 50% probability, Sentence A and Sentence B correspond to two consecutive sentences from the original corpus. Otherwise, a random sentence from the corpus is used. The combined length of the two “sentences” must be less than 512 tokens.
Masking procedure for each sentence:
15% of tokens are masked.
In 80% of cases, masked tokens are replaced by [MASK].
In 10% of cases, masked tokens are replaced by a different random token.
In the remaining 10%, masked tokens remain unchanged.

### Pretraining

The model was trained on 8 NVIDIA V100 GPUs (each with 16 GB memory) for 90 hours. Refer to the training code for detailed hyperparameters.

# Evaluation Results

When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:
|--|--|--|--|--|--|--|--|--|
Task |	MNLI |	QQP |	QNLI |	SST-2 |	CoLA |	STS-B |	MRPC |	RTE
 |82.2 |	88.5 |	89.2 |	91.3 |	51.3 |	85.8 |	87.5 |	59.9


# Limitations and Biases

While the training data for this model is generally neutral, it can still produce biased predictions. Additionally, it inherits some of the biases from its teacher model.

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/fill-mask-online-endpoint-oss)

# Evaluation samples

Evaluation type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)](https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/evaluation/fill-mask/fill-mask.ipynb)

# Sample inputs and outputs

#### Sample input
```json
{ 
  "input_data": ["Paris is [MASK] of France"]
} 
```

#### Sample output
```json
["part"]
```

DistilBERT, a transformers model, is designed to be smaller and quicker than BERT. It underwent pretraining on the same dataset in a self-supervised manner, utilizing the BERT base model as a reference. This entails training solely on raw texts, without human annotation, thus enabling the utilization of vast amounts of publicly accessible data. An automated process generates inputs and labels from these texts, guided by the BERT base model. Specifically, the pretraining process involved three objectives:

Distillation loss: The model was trained to produce probabilities akin to those of the BERT base model.
Masked language modeling (MLM): This constitutes a segment of the original training loss in the BERT base model. By randomly masking 15% of the words in a sentence, the model processes the entire masked sentence and endeavors to predict the masked words. This methodology differs from traditional recurrent neural networks (RNNs) or autoregressive models like GPT, which handle words sequentially or internally mask future tokens. MLM facilitates the acquisition of a bidirectional sentence representation by the model.
Cosine embedding loss: The model was also trained to generate hidden states that closely resemble those of the BERT base model.
In this manner, the model acquires a comparable internal representation of the English language to that of its teacher model, while being more efficient for inference or subsequent tasks.
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/distilbert-base-cased" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Training Details

### Preprocessing

The texts undergo lowercasing and tokenization using WordPiece with a vocabulary size of 30,000. The model's inputs follow the format:

[CLS] Sentence A [SEP] Sentence B [SEP]

In this structure, with a probability of 0.5, Sentence A and Sentence B represent two consecutive sentences from the original corpus. Otherwise, they correspond to another random sentence in the corpus. It's important to note that a "sentence" in this context refers to a consecutive span of text, often longer than a single sentence. The only constraint is that the combined length of the two "sentences" does not exceed 512 tokens.

The masking procedure for each sentence is detailed as follows:

15% of the tokens are masked.
In 80% of cases, the masked tokens are replaced by [MASK].
In 10% of cases, the masked tokens are replaced by a randomly selected token (different from the original).
In the remaining 10% of cases, the masked tokens remain unchanged.

### Pretraining
The model underwent training using 8 NVIDIA Tesla V100 GPUs, each with 16 GB of memory, for a duration of 90 hours. For specific details regarding hyperparameters and other training configurations, please refer to the training code.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-fill-mask" target="_blank">fill-mask-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-fill-mask" target="_blank">fill-mask-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-fill-mask" target="_blank">fill-mask-batch-endpoint.ipynb</a>| coming soon


### Evaluation Results

When fine-tuned on downstream tasks, this model demonstrates the following outcomes:

Glue test results:

Task	MNLI	QQP	QNLI	SST-2	CoLA	STS-B	MRPC	RTE
81.5	87.8	88.2	90.4	47.2	85.5	85.6	60.6

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

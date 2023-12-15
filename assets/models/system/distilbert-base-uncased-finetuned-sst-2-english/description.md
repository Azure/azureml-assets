**DistilBERT base uncased finetuned SST-2** model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned on SST-2.
This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).

This model can be used for  topic classification. You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. See the model hub to look for fine-tuned versions on a task that interests you.

# Training Details

## Training Data

The authors use the following Stanford Sentiment Treebank([sst2](https://huggingface.co/datasets/sst2)) corpora for the model.

## Training Procedure

### Fine-tuning hyper-parameters

- learning_rate = 1e-5
- batch_size = 32
- warmup = 600
- max_seq_length = 128
- num_train_epochs = 3.0

# Limitations and Biases

Based on a few experimentations, we observed that this model could produce biased predictions that target underrepresented populations.

For instance, for sentences like `This film was filmed in COUNTRY`, this binary classification model will give radically different probabilities for the positive label depending on the country (0.89 if the country is France, but 0.08 if the country is Afghanistan) when nothing in the input indicates such a strong semantic shift. In this [colab](https://colab.research.google.com/gist/ageron/fb2f64fb145b4bc7c49efc97e5f114d3/biasmap.ipynb), [Aurélien Géron](https://twitter.com/aureliengeron) made an interesting map plotting these probabilities for each country.

<img src="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/map.jpeg" alt="Map of positive probabilities per country." width="500"/>

We strongly advise users to thoroughly probe these aspects on their use-cases in order to evaluate the risks of this model. We recommend looking at the following bias evaluation datasets as a place to start: [WinoBias](https://huggingface.co/datasets/wino_bias), [WinoGender](https://huggingface.co/datasets/super_glue), [Stereoset](https://huggingface.co/datasets/stereoset).


# Sample inputs and outputs

### Sample input
```json
{
    "input_data": [
        "Today was an amazing day!",
        "It was an unfortunate series of events."
    ]
}
```

### Sample output
```json
[
  {
    "label": "POSITIVE",
    "score": 0.9998794794082642
  },
  {
    "label": "NEGATIVE",
    "score": 0.9995174407958984
  }
]
```

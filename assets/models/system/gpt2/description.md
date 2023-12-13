GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences.

This is the smallest version of GPT-2, with 124M parameters.

# Training Details

## Training Data

The OpenAI team wanted to train this model on a corpus as large as possible. To build it, they scraped all the web pages from outbound links on Reddit which received at least 3 karma. Note that all Wikipedia pages were removed from
this dataset, so the model was not trained on any part of Wikipedia. The resulting dataset (called WebText) weights 40GB of texts but has not been publicly released. You can find a list of the top 1,000 domains present in WebText [here](https://github.com/openai/gpt-2/blob/master/domains.txt).

## Preprocessing

The texts are tokenized using a byte-level version of Byte Pair Encoding (BPE) (for unicode characters) and a vocabulary size of 50,257. The inputs are sequences of 1024 consecutive tokens.

The larger model was trained on 256 cloud TPU v3 cores. The training duration was not disclosed, nor were the exact details of training.

# Evaluation Results

The model achieves the following results without any fine-tuning (zero-shot):

| Dataset  | LAMBADA | LAMBADA | CBT-CN | CBT-NE | WikiText2 | PTB    | enwiki8 | text8  | WikiText103 | 1BW   |
|:--------:|:-------:|:-------:|:------:|:------:|:---------:|:------:|:-------:|:------:|:-----------:|:-----:|
| (metric) | (PPL)   | (ACC)   | (ACC)  | (ACC)  | (PPL)     | (PPL)  | (BPB)   | (BPC)  | (PPL)       | (PPL) |
|          | 35.13   | 45.99   | 87.65  | 83.4   | 29.41     | 65.85  | 1.16    | 1,17   | 37.50       | 75.20 |

# Limitations and bias

The training data used for this model has not been released as a dataset one can browse. We know it contains a lot of unfiltered content from the internet, which is far from neutral. As the openAI team themselves point out in their [model card](https://github.com/openai/gpt-2/blob/master/model_card.md#out-of-scope-use-cases):

> Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases
> that require the generated text to be true.
>
> Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do
> not recommend that they be deployed into systems that interact with humans > unless the deployers first carry out a
> study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race,
> and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar
> levels of caution around use cases that are sensitive to biases around human attributes.

*Note: This bias will also affect all fine-tuned versions of this model.*

# Sample inputs and outputs

### Sample input
```json
{
    "input_data": [
        "I believe the meaning of life is"
    ],
    "params": {
        "top_p": 1.0,
        "temperature": 0.8,
        "max_new_tokens": 100,
        "do_sample": true,
        "return_full_text": true
    }
}
```

### Sample output
```json
[
  "I believe the meaning of life is to give way to you in the present moment to the things you love the most. We don't need to worry about your feelings of guilt, anger, or pain; we need to find ways to make things easier for you and help you get back to normal.\n\nAs a mother, I've always considered that the meaning of the world came from the love we gave each other. I believe that love is a life-sustaining energy that can help us reach our goal of one day"
]
```

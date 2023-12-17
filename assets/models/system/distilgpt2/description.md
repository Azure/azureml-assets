DistilGPT2 (short for Distilled-GPT2) is an English-language model pre-trained with the supervision of the 124 million parameter version of GPT-2. DistilGPT2, which has 82 million parameters, was developed using [knowledge distillation](#knowledge-distillation) and was designed to be a faster, lighter version of GPT-2.

Since DistilGPT2 is a distilled version of GPT-2, it is intended to be used for similar use cases with the increased functionality of being smaller and easier to run than the base model. 

The developers of GPT-2 state in their [model card](https://github.com/openai/gpt-2/blob/master/model_card.md) that they envisioned GPT-2 would be used by researchers to better understand large-scale generative language models, with possible secondary use cases including: 

> - *Writing assistance: Grammar assistance, autocompletion (for normal prose or code)*
> - *Creative writing and art: exploring the generation of creative, fictional texts; aiding creation of poetry and other literary art.*
> - *Entertainment: Creation of games, chat bots, and amusing generations.*

Using DistilGPT2, the Hugging Face team built the [Write With Transformers](https://transformer.huggingface.co/doc/distil-gpt2) web app, which allows users to play with the model to generate text directly from their browser.

# Training Details

## Training Data

DistilGPT2 was trained using [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/), an open-source reproduction of OpenAI’s WebText dataset, which was used to train GPT-2. See the [OpenWebTextCorpus Dataset Card](https://huggingface.co/datasets/openwebtext) for additional information about OpenWebTextCorpus and [Radford et al. (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) for additional information about WebText.

## Training Procedure

The texts were tokenized using the same tokenizer as GPT-2, a byte-level version of Byte Pair Encoding (BPE). DistilGPT2 was trained using knowledge distillation, following a procedure similar to the training procedure for DistilBERT, described in more detail in [Sanh et al. (2019)](https://arxiv.org/abs/1910.01108). 

# Evaluation Results

The creators of DistilGPT2 [report](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) that, on the [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) benchmark, GPT-2 reaches a perplexity on the test set of 16.3 compared to 21.1 for DistilGPT2 (after fine-tuning on the train set).


# Limitations and Biases

**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.**

As the developers of GPT-2 (OpenAI) note in their [model card](https://github.com/openai/gpt-2/blob/master/model_card.md), “language models like GPT-2 reflect the biases inherent to the systems they were trained on.” Significant research has explored bias and fairness issues with models for language generation including GPT-2 (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). 

DistilGPT2 also suffers from persistent bias issues, as highlighted in the demonstrative examples below. Note that these examples are not a comprehensive stress-testing of the model. Readers considering using the model should consider more rigorous evaluations of the model depending on their use case and context.

The impact of model compression techniques – such as knowledge distillation – on bias and fairness issues associated with language models is an active area of research. For example: 

- [Silva, Tambwekar and Gombolay (2021)](https://aclanthology.org/2021.naacl-main.189.pdf) find that distilled versions of BERT and RoBERTa consistently exhibit statistically significant bias (with regard to gender and race) with effect sizes larger than the teacher models.
- [Xu and Hu (2022)](https://arxiv.org/pdf/2201.08542.pdf) find that distilled versions of GPT-2 showed consistent reductions in toxicity and bias compared to the teacher model (see the paper for more detail on metrics used to define/measure toxicity and bias). 
- [Gupta et al. (2022)](https://arxiv.org/pdf/2203.12574.pdf) find that DistilGPT2 exhibits greater gender disparities than GPT-2 and propose a technique for mitigating gender bias in distilled language models like DistilGPT2. 

# Inference samples

Inference type|Python sample (Notebook)
|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/sdk-notebook-examples)
Real time|[text-generation-online-endpoint.ipynb](https://aka.ms/text-generation-online-endpoint-oss)

# Sample inputs and outputs

### Sample input
```json
{
    "input_data": [
        "I believe the meaning of life is"
    ],
    "params": {
        "top_p": 0.9,
        "temperature": 0.2,
        "max_new_tokens": 100,
        "do_sample": true,
        "return_full_text": true
    }
}
```

### Sample output
```json
[
  "I believe the meaning of life is not to be confused with the meaning of life.”"
]
```

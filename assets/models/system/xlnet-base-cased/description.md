## Model Description

# XLNet (base-sized model) 

XLNet model pre-trained on English language. It was introduced in the paper [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Yang et al. and first released in [this repository](https://github.com/zihangdai/xlnet/). 

Disclaimer: The team releasing XLNet did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.

## Intended uses & limitations

The model is mostly intended to be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?search=xlnet) to look for fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering. For tasks such as text generation, you should look at models like GPT2.

## Usage

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-1906-08237,
  author    = {Zhilin Yang and
               Zihang Dai and
               Yiming Yang and
               Jaime G. Carbonell and
               Ruslan Salakhutdinov and
               Quoc V. Le},
  title     = {XLNet: Generalized Autoregressive Pretraining for Language Understanding},
  journal   = {CoRR},
  volume    = {abs/1906.08237},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.08237},
  eprinttype = {arXiv},
  eprint    = {1906.08237},
  timestamp = {Mon, 24 Jun 2019 17:28:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1906-08237.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[fill-mask-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-fill-mask)|[fill-mask-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-fill-mask)
Batch |[fill-mask-batch-endpoint.ipynb](https://aka.ms/azureml-infer-batch-sdk-fill-mask)|coming soon

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Named Entity Recognition|[Conll2003](https://huggingface.co/datasets/conll2003)|[named-entity-recognition.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[named-entity-recognition.sh](https://aka.ms/azureml-ft-cli-token-classification)
Question Answering|Extractive Q&A|[SQUAD (Wikipedia)](https://huggingface.co/datasets/squad)|[extractive-qa.ipynb](https://aka.ms/azureml-ft-sdk-extractive-qa)|[extractive-qa.sh](https://aka.ms/azureml-ft-cli-extractive-qa)

### Model Evaluation samples

Task | Use case | Dataset | Python sample (Notebook) | CLI with YAML
|--|--|--|--|--|
Fill Mask|Fill Mask|[rcds/wikipedia-for-mask-filling](https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling)|[evaluate-model-fill-mask.ipynb](https://aka.ms/azureml-eval-sdk-fill-mask/)|[evaluate-model-fill-mask.yml](https://aka.ms/azureml-eval-cli-fill-mask/)

### Sample inputs and outputs

#### Sample input
```json
{'input_data': {'input_string': ['My name is John and I am', 'Once upon a time,'], 'parameters': {'max_new_tokens': 100, 'do_sample': True, 'temperature': 0.5, 'top_p': 0.5}}}
```

#### Sample output
```json
[{"0": "My name is John and I am a \"new\" \"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\""}, {"0": "Once upon a time, a man called Rasputin was a priest. He was a priest of the Church of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the Holy Trinity. He was a priest of the"}]
```

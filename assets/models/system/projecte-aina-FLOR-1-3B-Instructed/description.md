## Model Description


# FLOR-1.3B Instructed

## Table of Contents
<details>
<summary>Click to expand</summary>

- [Model description](#model-description)
- [Intended uses and limitations](#intended-uses-and-limitations)
- [How to use](#how-to-use)
- [Limitations and bias](#limitations-and-bias)
- [Training](#training)
- [Evaluation](#evaluation)
- [Additional information](#additional-information)

</details>

## Model description

**FLOR-1.3B-Instructed** is a 1.3B-parameter transformer-based causal language model for Catalan, Spanish, and English, trained on a combined dataset from [InstruCat](https://huggingface.co/datasets/BSC-LT/InstruCat), a Catalan language set of instruction generated automatically from prject-aina task orientated dataset, a subset of the [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset for English, and [MENTOR_ES](https://huggingface.co/datasets/projecte-aina/MENTOR_ES) and [MENTOR_CA](https://huggingface.co/datasets/projecte-aina/MENTOR_CA), a Spanish and Catalan sets of instructions commisioned by the BSC Language Technologies Unit. 
It is th result of a language adaptation technique performed on [BLOOM-7.1B](https://huggingface.co/bigscience/bloom-7b1), 
which involves modifying the model's vocabulary and embedding layer, and continuously pre-training the model with 140B tokens in our target languages.
Blog post describing the base model with more parameters: [flor-6-3b, a chinchilla compliant model](https://medium.com/@mpamies247/flor-6-3b-a-chinchilla-compliant-model-for-catalan-spanish-and-english-7cdb389a9aac)

## Intended uses and limitations

The **FLOR-1.3B-Instructed** model is ready-to-use for some downstream tasks. 
It can perform text-generation tasks because fine-tuned for specific scenarios, such as summarization, Question Answering, creative writing, etc.

## How to use
```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="projecte-aina/FLOR-1.3B-Instructed")

instruction = "Quants habitants té Mataró?"

context = "Mataró és una ciutat de Catalunya, capital de la comarca del Maresme. Situada al litoral mediterrani, a uns 30 km al nord-est de Barcelona, ha estat tradicionalment un centre administratiu de rellevància territorial i un pol de dinamisme econòmic. Compta amb prop de 130.000 habitants, essent actualment la vuitena població del Principat i la tretzena dels Països Catalans. "

# We need to format the prompt and context using ### and \n

def givePrediction(instruction, context, max_new_tokens=50, repetition_penalty=1.2, top_k=50, top_p=0.95, do_sample=True, temperature=0.5)
    text = f"### Instruction\n{{instruction}}\n### Context\n{{context}}\n### Answer\n"
    response = pipe(text.format(instruction=instruction, context=context),temperature=temperature,repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,top_k=top_k, top_p=top_p, do_sample=do_sample)[0]["generated_text"]
    answer = response.split("###")[-1][8:-1]
    return answer

answer = givePrediction(instruction, context)

print(answer)
'130 000'

```

## Limitations and bias
At the time of submission, no measures have been taken to estimate the bias and toxicity embedded in the model. 
However, we are well aware that our models may be biased since the corpora have been collected using crawling techniques 
on multiple web sources. We intend to conduct research in these areas in the future, and if completed, this model card will be updated. 


## Training


### Instruction Data

The training corpus is composed of 140B tokens gathered from web crawlings and public domain data.

## Additional information

### Author
The Language Technologies Unit from Barcelona Supercomputing Center.

### Contact
For further information, please send an email to <langtech@bsc.es>.

### Copyright
Copyright(c) 2023 by Language Technologies Unit, Barcelona Supercomputing Center.

### License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### Funding
This work was funded by [Departament de la Vicepresidència i de Polítiques Digitals i Territori de la Generalitat de Catalunya](https://politiquesdigitals.gencat.cat/ca/inici/index.html#googtrans(ca|en) within the framework of [Projecte AINA](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina).

### Disclaimer

<details>
<summary>Click to expand</summary>

The model published in this repository is intended for a generalist purpose and is available to third parties under a permissive Apache License, Version 2.0. 

Be aware that the model may have biases and/or any other undesirable distortions.

When third parties deploy or provide systems and/or services to other parties using this model (or any system based on it) 
or become users of the model, they should note that it is their responsibility to mitigate the risks arising from its use and, 
in any event, to comply with applicable regulations, including regulations regarding the use of Artificial Intelligence.

In no event shall the owner and creator of the model (Barcelona Supercomputing Center) 
be liable for any results arising from the use made by third parties.

</details>

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
{'input_data': {'input_string': ['My name is John and I am', 'Once upon a time,'], 'parameters': {'max_new_tokens': 25, 'do_sample': True, 'temperature': 0.5, 'top_p': 0.5}}}
```

#### Sample output
```json
[{"0": "My name is John and I am a man who has been living in the desert for a long time. I am a man who has been living in the desert"}, {"0": "Once upon a time, the world was inhabited by a race of elves called the Elves of the Forest. They lived in harmony"}]
```

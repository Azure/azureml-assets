## Model Description

# FLOR-6.3B

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

**FLOR-6.3B** is a 6.3B-parameter transformer-based causal language model for Catalan, Spanish, and English. 
It is the result of a language adaptation technique performed on [BLOOM-7.1B](https://huggingface.co/bigscience/bloom-7b1), 
which involves modifying the model's vocabulary and embedding layer, and continuously pre-training the model with 140B tokens in our target languages.

For more details, take a look at [this blogpost](https://medium.com/@mpamies247/flor-6-3b-a-chinchilla-compliant-model-for-catalan-spanish-and-english-7cdb389a9aac) about the project.

## Intended uses and limitations

The **FLOR-6.3B** model is ready-to-use only for causal language modeling. 
It can perform text-generation tasks and be fine-tuned for specific scenarios.

## How to use
```python
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

input_text = "Sovint em trobo pensant en tot allò que"

model_id  = "projecte-aina/FLOR-6.3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
generation = generator(
    input_text,
    do_sample=True,
    top_k=10,
    eos_token_id=tokenizer.eos_token_id,
)

print(f"Result: {generation[0]['generated_text']}")
```

## Limitations and bias
At the time of submission, no measures have been taken to estimate the bias and toxicity embedded in the model. 
However, we are well aware that our models may be biased since the corpora have been collected using crawling techniques 
on multiple web sources. We intend to conduct research in these areas in the future, and if completed, this model card will be updated. 


## Training

### Language adaptation and training

The language adaptation technique used to create FLOR-6.3B requires the vocabulary of the source model 
to be adapted before continuing its pre-training with data in the target languages. Specifically, we proceeded as follows:
1) We trained our own BPE tokenizer for Catalan, Spanish, and English, and replaced the original BLOOM tokenizer and vocabulary with it. This procedure implied a downsizing of the original BLOOM's embedding layer and, therefore, a model compression from 7.1B parameters to 6.3B.
2) The embeddings corresponding to tokens that are present in both the original and the target vocabulary (matching tokens) were used for initialization.
3) The embeddings from tokens not present in BLOOM's original vocabulary were initialized as the average of all embeddings.
4) The model was initialized with the weights from BLOOM-7.1B, and with our adapted tokenizer (step 1) and embeddings (steps 2-3).
5) The model was then trained on a corpus that contains a mixture of Catalan, Spanish, and English data.

### Training data

The training corpus is composed of 140B tokens gathered from web crawlings and public domain data. Most of the sources in Catalan have been obtained from the [CATalog 1.0](https://huggingface.co/datasets/projecte-aina/CATalog) dataset, filtered with a minimum threshold of 0.6 and oversampling some of the sources it integrates to different extents. 

Dataset	| Language	| Words (per-epoch)	| Epochs	| Total Tokens |
|---------------------|----------|--------------------|--------------|--------------|
mc4	| ca	| 5,861.79M	| 1.5	| 13,452.81M |
MaCoCu	| ca	| 1,658.89M	| 2	| 5,076.21M |
CaWac	| ca	| 1,286.83M	| 2.5	| 4,922.14M |
oscar-2301	| ca	| 1,784.57M	| 1.75	| 4,778.17M |
RacoCatala Articles	| ca	| 358.57M	| 4	| 2,194.42M |
RacoCatala Forums	| ca	| 1,301.12M	| 1	| 1,990.71M |
Tesis (TDX)	| ca	| 323.60M	| 4	| 1,980.46M |
oscar-2201	| ca	| 1,155.35M	| 1	| 1,767.69M |
Wikipedia	| ca	| 266.69M	| 4	| 1,632.17M |
Nació Digital	| ca	| 216.27M	| 4	| 1,323.59M |
colossal-oscar-05-06-23	| ca	| 207.59M	| 4	| 1,270.43M |
colossal-oscar-03-04-23	| ca	| 195.43M	| 4	| 1,196.01M |
colossal-oscar-2022-27	| ca	| 195.03M	| 4	| 1,193.59M |
Crawling populars	| ca	| 683.25M	| 1	| 1,045.38M |
El Món	| ca	| 85.27M	| 4	| 521.85M |
ACN	| ca	| 81.25M	| 4	| 497.22M |
DOGV	| ca	| 76.48M	| 4	| 468.05M |
DOGC	| ca	| 70.51M	| 4	| 431.51M |
Vilaweb	| ca	| 46.90M	| 4	| 287.04M |
hplt	| ca	| 160.27M	| 1	| 245.21M |
Les Corts Valencianes	| ca	| 26.88M	| 4	| 164.53M |
IB3	| ca	| 15.82M	| 4	| 96.82M |
BOUA	| ca	| 13.42M	| 4	| 82.13M |
Parlament	| ca	| 10.09M	| 4	| 61.77M |
Aquí Berguedà	| ca	| 8.23M	| 4	| 50.34M |
Wikimedia	| ca	| 3.90M	| 4	| 23.88M |
Gutenberg	| ca	| 1.29M	| 4	| 7.87M |
OSCAR 23.01	| es	| 53,244.56M	| 0.303	| 23,070.34M |
colossal_oscar_05-06-23	| es	| 5,548.27M	| 1	| 7,934.02M |
colossal_oscar_03-04-23	| es	| 5,090.46M	| 1	| 7,279.36M |
All_bio_corpora	| es	| 954.85M	| 2	| 2,730.88M |
Wikipedia	| es	| 777.49M	| 2	| 2,223.63M |
BOE	| es	| 1,031.28M	| 1	| 1,474.73M |
Tesis (TDX)	| es	| 268.66M	| 2	| 768.37M |
Eurlex	| es	| 459.19M	| 1	| 656.64M |
CSIC	| es	| 156.76M	| 2	| 448.33M |
BORME	| es	| 63.23M	| 1	| 90.42M |
colossal_oscar_05-06-23	| en	| 51,615.35M	| 0.25	| 21,162.30M |
colossal_oscar_03-04-23	| en	| 49,454.01M	| 0.14	| 11,354.64M |
Wikipedia	| en	| 2,116.53M	| 2	| 6,942.23M |
Gutenberg	| en	| 3,513.82M	| 1	| 5,762.66M |
Eurlex	| en	| 438.92M	| 1	| 719.83M |
legal-mc4	| en	| 417.97M	| 1	| 685.47M |

### Languages

The training data has the same amount of Catalan, Spanish, and English texts. 
The table below shows the final language distribution:

|Language|Percentage|
|--------|----------|
|   Catalan (CA)   |  33.39%  |
|   Spanish (ES)   |  33.32%  |
|   English (EN)   |  33.29%  |

### Framework
The training was conducted in 16 Cerebras' [CS-2 systems](https://www.cerebras.net/product-system/) 
using the [cs-2.0.2](https://github.com/Cerebras/modelzoo/releases/tag/Release_2.0.2) release of their software.

## Evaluation
FLOR-6.3B has been evaluated in a 5-shot setting, using EleutherAI's *LM Evaluation Harness*. 
The evaluation benchmark includes tasks in Catalan, Spanish, and English, with particular emphasis on Catalan datasets.

The tasks were chosen to cover several evaluation areas in order to provide a comprehensive overview of the model's capabilities. 
The baselines used to compare our results are multilingual and English open-source 7B models and smaller models of the FLOR family of models: **TBC**.

Our implementation of EleutherAI's *LM Evaluation Harness* can be found [here](https://github.com/langtech-bsc/lm-evaluation-harness/tree/FLOR-eval).

The following is a list of evaluation areas and their respective datasets:
- Reading Comprehension: [Belebele](https://huggingface.co/datasets/facebook/belebele)
- Question Answering: [XQuAD](https://huggingface.co/datasets/xquad), [CatalanQA](https://huggingface.co/datasets/projecte-aina/catalanqa), [CoQCat](https://huggingface.co/datasets/projecte-aina/CoQCat)
- Natural Language Inference: [XNLI](https://huggingface.co/datasets/xnli) and its translation to Catalan ([XNLI-ca](https://huggingface.co/datasets/projecte-aina/xnli-ca)), [TE-ca](https://huggingface.co/datasets/projecte-aina/teca)
- Paraphrase Identification: [PAWS-X](https://huggingface.co/datasets/paws-x) and its translation to Catalan ([PAWS-ca](https://huggingface.co/datasets/projecte-aina/PAWS-ca)), [Parafraseja](https://huggingface.co/datasets/projecte-aina/Parafraseja)
- Commonsense Reasoning: [COPA](https://people.ict.usc.edu/~gordon/copa.html) and its translation to Catalan ([COPA-ca](https://huggingface.co/datasets/projecte-aina/COPA-ca))
- Translation: [Flores-200](https://huggingface.co/datasets/Muennighoff/flores200)

### Results

| Dataset     |  Lang. |          Task             |  FLOR-6.3B  |  BLOOM-7.1B |
|-------------|--------|----------------------------|-------------|-------------|
| Teca        |   ca   | Natural Language Inference | **49.79**🔥 | 46.91       |
| XNLI        |   ca   | Natural Language Inference | **51.70**🔥 | 49.20       |
| XNLI        |   es   | Natural Language Inference | **50.28**🔥 | 47.62       |
| XNLI        |   en   | Natural Language Inference | **52.55**🔥 | 51.96       |
| Belebele    |   ca   | Reading Comprehension      | **48.98**🔥 | 48.57       |
| Belebele    |   es   | Reading Comprehension      | **48.16**   | **48.16**   |
| Belebele    |   en   | Reading Comprehension      | 49.80       | **50.20**🔥 |
| CatalanQA   |   ca   | Question Answering         | **71.80**🔥 | 69.54       |
| CoQCat      |   ca   | Question Answering         | **65.96**🔥 | 58.49       |
| XQuAD       |   ca   | Question Answering         | 59.01       | **60.94**🔥 |
| XQuAD       |   es   | Question Answering         | **63.80**🔥 | 61.76       |
| XQuAD       |   en   | Question Answering         | **70.02**🔥 | 69.76       |
| COPA        |   ca   | Question Answering         | **78.00**🔥 | 72.60       |
| COPA        |   en   | Question Answering         | **81.00**🔥 | 79.00       |
| XStoryCloze |   es   | Question Answering         | **69.82**🔥 | 66.45       |
| XStoryCloze |   en   | Question Answering         | **74.45**🔥 | 70.81       |
| Parafraseja |   ca   | Paraphrase Identification  | **62.88**🔥 | 60.27       |
| PAWS-X      |   ca   | Paraphrase Identification  | **59.70**🔥 | 59.35       |
| PAWS-X      |   es   | Paraphrase Identification  | 57.70       | **58.65**🔥 |
| PAWS-X      |   en   | Paraphrase Identification  | 59.65       | **62.85**🔥 |
| FLoRes      | ca->es | Machine Translation        | **24.98**🔥 | 24.21       |
| FLoRes      | es->ca | Machine Translation        | **25.24**🔥 | 23.19       |
| FLoRes      | ca->en | Machine Translation        | **42.89**🔥 | 40.93       |
| FLoRes      | en->ca | Machine Translation        | **39.29**🔥 | 34.30       |
| FLoRes      | es->en | Machine Translation        | **28.61**🔥 | 27.48       |
| FLoRes      | en->es | Machine Translation        | **25.35**🔥 | 23.72       |

Note: The metrics are F1-score for question-answering tasks, BLEU for translation, and accuracy for the rest.

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
Real time|[text-generation-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-text-generation)|[text-generation-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-text-generation)
Batch |[	text-generation-batch-endpoint.ipynb](https://aka.ms/azureml-infer-batch-sdk-text-generation)|coming soon

### Sample inputs and outputs

#### Sample input
```json
{
  "input_data": {
    "input_string": [
      "Once upon a time,"
    ],
    "parameters": {
      "top_p": 0.8,
      "temperature": 0.8,
      "max_new_tokens": 90,
      "do_sample": true
    }
  }
}
```

#### Sample output
```json
[
  {
    "0": "Once upon a time, there was a village where the villagers lived in peace and harmony. They worked together, shared their food and resources, and lived in a way that made them happy.\n\nOne day, a stranger arrived in the village. He was a wise and powerful man who could see the future. He told the villagers that their way of life was not sustainable and that they needed to change it.\n\nThe villa"
  }
]
```

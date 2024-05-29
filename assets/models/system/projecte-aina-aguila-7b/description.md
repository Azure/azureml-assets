## Model Description

# Aguila-7b

## Table of Contents
<details>
<summary>Click to expand</summary>

- [Model description](#model-description)
- [Intended uses and limitations](#intended-uses-and-limitations)
- [How to use](#how-to-use)
- [Limitations and bias](#limitations-and-bias)
- [Training](#training)
- [Additional information](#additional-information)

</details>

## Model description

**Ǎguila-7B** is a transformer-based causal language model for Catalan, Spanish, and English. It is based on the Falcon-7B model and has been trained on a 26B token trilingual corpus collected from publicly available corpora and crawlers.

For more details, take a look at [this blogpost](https://medium.com/@mpamies247/flor-6-3b-a-chinchilla-compliant-model-for-catalan-spanish-and-english-7cdb389a9aac) about the project.

More information available in the following post from Medium.com: [Introducing Ǎguila, a new open-source LLM for Spanish and Catalan](https://medium.com/@mpamies247/introducing-a%CC%8Cguila-a-new-open-source-llm-for-spanish-and-catalan-ee1ebc70bc79)

## Intended uses and limitations

The **Ǎguila-7B** model is ready-to-use only for causal language modeling to perform text-generation tasks. However, it is intended to be fine-tuned for downstream tasks.

## How to use
```python
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

input_text = "El mercat del barri és fantàstic, hi pots trobar"

model_id  = "projecte-aina/aguila-7b"
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
At the time of submission, no measures have been taken to estimate the bias and toxicity embedded in the model. However, we are well aware that our models may be biased since the corpora have been collected using crawling techniques on multiple web sources. We intend to conduct research in these areas in the future, and if completed, this model card will be updated.


## Training

### Language adaptation and training

The original Falcon-7B model was adapted to Spanish and Catalan by swapping the tokenizer and adjusting the embedding layer.

### Training data

The training corpus consists of 26B tokens of several corpora gathered from web crawlings and public domain data.

| Dataset             | Language | Words (per-epoch) | Epochs       |
|---------------------|----------|--------------------|--------------|
| Wikipedia           | en       |           2169.97M |  1.428144485 |
| C4_es               | es       |          53709.80M | 0.1049686196 |
| Biomedical          | es       |            455.03M | 0.7140722425 |
| Legal               | es       |            995.70M | 0.7140722425 |
| Wikipedia           | es       |            693.60M |  1.428144485 |
| Gutenberg           | es       |             53.18M | 0.7140722425 |
| C4_ca               | ca       |           2826.00M |  2.142216727 |
| Biomedical          | ca       |             11.80M |  1.428144485 |
| RacoCatalà Noticias | ca       |             17.16M |  2.142216727 |
| RacoCatalà Forums   | ca       |            333.73M |  2.142216727 |
| CaWaC               | ca       |             57.79M |  2.142216727 |
| Wikipedia           | ca       |            228.01M |  3.570361212 |
| Vilaweb             | ca       |             50.34M |  2.142216727 |

The dataset has the following language distribution:

| Language             | Percentage |
|----------------------|------------|
| En                   | 16.84%     |
| Es                   | 41.38%     |
| Ca                   | 41.79%     |


### Languages

The training data has the same amount of Catalan and Spanish texts, and a smaller amount of English data. 
The table below shows the final language distribution:

|Language|Percentage|
|--------|----------|
|   English (EN)   |  16.84%  |
|   Spanish (ES)   |  41.38%  |
|   Catalan (CA)   |  41.79%  |

Note: A small amount of English data was kept to avoid catastrophic forgetting.

### Training procedure

The training corpus has been tokenized using a byte version of Byte-Pair Encoding (BPE) with a vocabulary size of 50,257 tokens. After training a new tokenizer and adapting falcon-7b's embedding layer, the model was further pre-trained in three target languages: Catalan, Spanish and English.

The training lasted a total of 320 hours on 8 NVIDIA H100 GPUs with 80GB RAM.

### Training hyperparameters

- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- train_batch_size: 1
- eval_batch_size: 1
- total_train_batch_size: 8
- total_eval_batch_size: 8
- optimizer: Adam
- betas: (0.9,0.999)
- epsilon: 1e-08
- learning_rate: 5e-05
- lr_scheduler_type: linear
- num_epochs: 1.0

### Framework versions

- Pytorch 2.0.0
- Transformers 4.30.2
- Datasets 2.13.1
- Tokenizers 0.13.3

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

The [Spanish State Secretariat for Digitalization and Artificial Intelligence](https://portal.mineco.gob.es/en-us/digitalizacionIA/Pages/sedia.aspx) within the framework of the [Plan de Impulso de las Tecnologías del Lenguaje.](https://plantl.mineco.gob.es/Paginas/index.aspx)

### Disclaimer

The model published in this repository is intended for a generalist purpose and is available to third parties under a permissive Apache License, Version 2.0. 

Be aware that the model may have biases and/or any other undesirable distortions.

When third parties deploy or provide systems and/or services to other parties using this model (or any system based on it) 
or become users of the model, they should note that it is their responsibility to mitigate the risks arising from its use and, 
in any event, to comply with applicable regulations, including regulations regarding the use of Artificial Intelligence.

In no event shall the owner and creator of the model (Barcelona Supercomputing Center) 
be liable for any results arising from the use made by third parties.


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
    "0": "Once upon a time, the person who owned the house would be in charge of cleaning it and of cooking. After the owner left, the kitchen would be dismantled and stored away.\n\nWhen the kitchen was complete, the family would go to the front of the house and take a seat under the porch. The porch would be covered with a blanket, and the family would sit and eat while the k"
  }
]
```

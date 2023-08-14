CamemBERT is a state-of-the-art language model for French developed by a team of researchers. It is based on the RoBERTa model and is available in 6 different versions on Hugging Face. It can be used for fill-in-the-blank tasks. However, it has been pretrained on a subcorpus of OSCAR which may contain lower quality data and personal and sensitive information. Also, there may be biases and historical stereotypes present in the model. The model is licensed under the MIT license, and more information can be found in the research paper and on the Camembert website. It was trained on the OSCAR dataset, which is a multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the Ungoliant architecture.
The "Camembert" model is an advanced computer program designed for understanding and processing human language, a field known as natural language processing (NLP). It's based on a powerful architecture called a transformer, which is very good at handling language-related tasks. Camembert is particularly optimized for the French language and can be used to analyze text, understand context, and perform tasks like translation, sentiment analysis, and more. It's a tool that helps computers understand and work with human language in a sophisticated way.

It serves as the starting point for customization and fine-tuning for specific tasks, such as sentiment analysis, question answering, or text generation. This base model already possesses a general understanding of language, making it easier to adapt to more specific purposes.

<br>Please Note: This model accepts masks in `<mask>` format. See Sample input for reference. 
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/camembert-base" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

|      Model                             | #Params          |  Arch     |
|----------------------------------------|------------------|-----------|
|camembert-base                          |   110M           |  Base     |
|camembert/camembert-large               |   335M           |  Large    |
|camembert/camembert-base-ccnet          |   110M           |  Base     |
|camembert/camembert-base-wikipedia-4gb  |   110M           |  Base     |
|camembert/camembert-base-oscar-4gb      |   110M           |  Base     |
|camembert/camembert-base-ccnet-4gb      |   110M           |  Base     |

### Authors
CamemBERT was trained and evaluated by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
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

Task|Use case|Python sample (Notebook)|CLI with YAML
|--|--|--|--|
Fill Mask|Fill Mask|<a href="https://huggingface.co/datasets/rcds/wikipedia-for-mask-filling" target="_blank">rcds/wikipedia-for-mask-filling</a>|<a href="https://aka.ms/azureml-eval-sdk-fill-mask/" target="_blank">evaluate-model-fill-mask.ipynb</a>|<a href="https://aka.ms/azureml-eval-cli-fill-mask/" target="_blank">evaluate-model-fill-mask.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["Paris is the <mask> of France.", "Today is a <mask> day!"]
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

The MiniLM-L12-H384-uncased model is a microsoft language model for extractive question answering in English. It was trained on the SQuAD 2.0 dataset and has been evaluated on the SQuAD 2.0 dev set with the official eval script. The model's performance results were an exact match of 76.13 and F1 score of 79.50. The model can be used with Transformers, FARM, or haystack. The model was developed by Vaishali Pal, Branden Chan, Timo Möller, Malte Pietsch, and Tanay Soni, who are employees of deepset, a company focused on bringing NLP to the industry via open source.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[sdk-example.ipynb](https://aka.ms/azureml-infer-sdk)|[cli-example.sh](https://aka.ms/azureml-infer-cli)
Batch | todo


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Question Answering||[Squad v2](https://huggingface.co/datasets/squad_v2)|[evaluate-model-question-answering.ipynb](https://aka.ms/azureml-eval-sdk-question-answering)|

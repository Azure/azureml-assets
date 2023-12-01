Automated Machine Learning, or AutoML, is a process that automates the repetitive and time-consuming tasks involved in developing machine learning models. This helps data scientists, analysts, and developers to create models more efficiently and with higher quality, resulting in increased productivity and scalability.

AutoML Named entity recognition (NER) is a sub-task of information extraction (IE) that seeks out and categorizes specified entities in a body or bodies of texts. NER is also known simply as entity identification, entity chunking and entity extraction.

With this functionality, you can:
* Directly use datasets coming from [Azure Machine Learning data labeling](https://learn.microsoft.com/azure/machine-learning/how-to-create-text-labeling-projects?view=azureml-api-2)
* Utilize labeled data to create NLP models without any training code.
* Enhance model performance by selecting the appropriate algorithm and fine-tuning the hyperparameters selecting the appropriate algorithm from a [large selection of models](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=cli#supported-model-algorithms) or let AutoML find the best model for you.
* Either download or deploy the resulting model as a endpoint in Azure Machine Learning.
* Scale the operationalization process with the help of Azure Machine Learning's MLOps and ML Pipelines capabilities.

See [How to train nlp models](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=cli) for more information.

### Documentation

#### Prepare Data
To create NLP models, it is necessary to provide labeled text data as input for model training. Named entity recognition requires CoNLL format. The file must contain exactly two columns and in each row, the token and the label is separated by a single space. Example as below:

```txt
Hudson B-loc
Square I-loc
is O
a O
famous O
place O
in O
New B-loc
York I-loc
City I-loc

Stephen B-per
Curry I-per
got O
three O
championship O
rings O
```

Please see [documentation](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#named-entity-recognition-ner) for data requirements.

#### Language Setting

Currently, language selection defaults to English. But Automated ML supports 104 languages leveraging language specific and multilingual pre-trained text DNN models. Please see [Language setting](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#language-settings) for documentation.

#### Train a Model

You can initiate [individual trials](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=cli#data-validation), or perform a [manual sweeps](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#model-sweeping-and-hyperparameter-tuning-preview), which explores multiple hyperparameter values near the more promising models and hyperparameter configurations. 

For more information, see [Model sweeping and hyperparameter tuning](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#model-sweeping-and-hyperparameter-tuning-preview).

### Code samples


Task|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|
Named Entity Recognition|[CoNLL-2003](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion?select=valid.txt)|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-nlp-text-named-entity-recognition-task-distributed-sweeping/automl-nlp-text-ner-task-distributed-with-sweeping.ipynb" target="_blank">automl-nlp-text-ner-task-distributed-with-sweeping.ipynb</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/cli/jobs/automl-standalone-jobs/cli-automl-text-ner-conll-distributed-sweeping/cli-automl-text-ner-conll2003-distributed-sweeping.yml" target="_blank">cli-automl-text-ner-conll2003-distributed-sweeping.yml
</a>



### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
    "input_data": {
        ["Stephen\nCurry\nis\na\nsuper\nstar!"]
    }
}
```

#### Sample output

```json
["Stephen B-PER Curry I-PER is O a O super O star! O"]
```
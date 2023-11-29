Automated Machine Learning, or AutoML, is a process that automates the repetitive and time-consuming tasks involved in developing machine learning models. This helps data scientists, analysts, and developers to create models more efficiently and with higher quality, resulting in increased productivity and scalability.

AutoML Text Classification enables you to classify or categorize texts into predefined groups. Your dataset should be a labeled set of texts with their relevant tags that categorize each piece of text into a predefined group.

With this functionality, you can:
* Directly use datasets coming from [Azure Machine Learning data labeling](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-text-labeling-projects?view=azureml-api-2)
* Utilize labeled data to create NLP models without any training code.
* Enhance model performance by selecting the appropriate algorithm and fine-tuning the hyperparameters selecting the appropriate algorithm from a [large selection of models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=cli#supported-model-algorithms) or let AutoML find the best model for you.
* Either download or deploy the resulting model as a endpoint in Azure Machine Learning.
* Scale the operationalization process with the help of Azure Machine Learning's MLOps and ML Pipelines capabilities.

See [How to train nlp models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=cli) for more information.

### Documentation

#### Prepare Data
To create NLP models, it is necessary to provide labeled text data as input for model training. For text classification, the dataset can contain several text columns and exactly one label column. 

Please see [documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#preparing-data) for data preparation requirements.

#### Language Setting

Currently, language selection defaults to English. But Automated ML supports 104 languages leveraging language specific and multilingual pre-trained text DNN models. Please see [langugage setting](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#language-settings) for documentation.

#### Train a Model

You can initiate [individual trials](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=cli#data-validation), or perform a [manual sweeps](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#model-sweeping-and-hyperparameter-tuning-preview), which explores multiple hyperparameter values near the more promising models and hyperparameter configurations. 

For more information, see [Model sweeping and hyperparameter tuning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2&tabs=python#model-sweeping-and-hyperparameter-tuning-preview).

### Code samples


Task|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|
Multiclass Text Classification|[Yelp review](https://huggingface.co/datasets/yelp_review_full)|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-nlp-text-classification-multiclass-task-sentiment-analysis/automl-nlp-multiclass-sentiment-mlflow.ipynb" target="_blank">automl-nlp-multiclass-sentiment-mlflow.ipynb</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/cli/jobs/automl-standalone-jobs/cli-automl-text-classification-newsgroup/cli-automl-text-classification-newsgroup.yml" target="_blank">cli-automl-text-classification-newsgroup.yml</a>
Multilabel Text Classification|[arXiv paper abstract](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts)|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-nlp-text-classification-multilabel-task-paper-categorization/automl-nlp-multilabel-paper-cat.ipynb" target="_blank">automl-nlp-multilabel-paper-cat.ipynb</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/cli/jobs/automl-standalone-jobs/cli-automl-text-classification-multilabel-paper-cat/cli-automl-text-classification-multilabel-paper-cat.yml" target="_blank">cli-automl-text-classification-multilabel-paper-cat.yml</a>



### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
    "input_data": {
        "input_string": ["Today was an amazing day!", "It was an unfortunate series of events."]
    }
}
```

#### Sample output

```json
[
    {
        "0": "Fake"
    },
    {
        "0": "Fake"
    }
]
```
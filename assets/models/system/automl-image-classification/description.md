Automated Machine Learning, or AutoML, is a process that automates the repetitive and time-consuming tasks involved in developing machine learning models. This helps data scientists, analysts, and developers to create models more efficiently and with higher quality, resulting in increased productivity and scalability.
AutoML Image Classification enables you to train machine learning models to perform *image multiclass* and *image multilabel* tasks according to your own defined labels. It is a computer vision task that involves analyzing and categorizing an image into specific label(s).

With this functionality, you can:
* Directly use datasets coming from [Azure Machine Learning data labeling](https://learn.microsoft.com/azure/machine-learning/how-to-create-image-labeling-projects?view=azureml-api-2)
* Utilize labeled data to create image models without any training code.
* Enhance model performance by selecting the appropriate algorithm and fine-tuning the hyperparameters selecting the appropriate algorithm from a [large selection of models](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=cli#supported-model-architectures) or let AutoML find the best model for you.
* Either download or deploy the resulting model as a endpoint in Azure Machine Learning.
* Scale the operationalization process with the help of Azure Machine Learning's MLOps and ML Pipelines capabilities.

See [How to train image models](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=cli) for more information.

### Documentation

#### Prepare Data
To create computer vision models, it is necessary to provide labeled image data as input for model training. This data needs to be in the form of an MLTable, which can be created from training data in JSONL format. Please see [documentation](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=python#jsonl-schema-samples) for JSONL Schema and consuming the same in MLTable.

#### Train a Model

You can initiate [individual trials](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=python#individual-trials), [manual sweeps](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=python#manually-sweeping-model-hyperparameters), or [automatic sweeps](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=python#automatically-sweeping-model-hyperparameters-automode). It is suggested to begin with an automatic sweep to establish a baseline model. Afterward, you can experiment with individual trials using specific models and hyperparameter configurations. Lastly, manual sweeps can be used to explore multiple hyperparameter values near the more promising models and hyperparameter configurations. This three-step process (automatic sweep, individual trials, manual sweeps) helps avoid searching the entirety of the hyperparameter space, which grows exponentially with the number of hyperparameters.

For more information, see [how to configure experiments](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=python#configure-experiments)

### Code samples

Task|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|
Image Multi-class classification|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip)|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-image-classification-multiclass-task-fridge-items/automl-image-classification-multiclass-task-fridge-items.ipynb" target="_blank">image-classification-multiclass.ipynb</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/cli/jobs/automl-standalone-jobs/cli-automl-image-classification-multiclass-task-fridge-items/cli-automl-image-classification-multiclass-task-fridge-items.yml" target="_blank">image-classification-multiclass.yml</a>
Image Multi-label classification|[multilabel fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip)|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-image-classification-multilabel-task-fridge-items/automl-image-classification-multilabel-task-fridge-items.ipynb" target="_blank">image-classification-multilabel.ipynb</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/cli/jobs/automl-standalone-jobs/cli-automl-image-classification-multilabel-task-fridge-items/cli-automl-image-classification-multilabel-task-fridge-items.yml" target="_blank">image-classification-multilabel.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
   "input_data":{
      "columns":[
         "image"
      ],
      "index":[0, 1],
      "data":[
         ["image1"],
         ["image2"]
      ]
   }
}
```
Note:
- "image1" and "image2" should be strings in `base64` format.

#### Sample output

```json
[
    {
        "probs": [0.95, 0.03],
        "labels": ["label1", "label2"]
    },
    {
        "probs": [0.04, 0.93],
        "labels": ["label1", "label2"]
    }
]
```

#### Model inference - visualization
For a sample image below, the top 3 labels are 'African elephant, Loxodonta africana', 'tusker', 'Indian elephant, Elephas maximus'.

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_automl_image_classification.png" alt="image classification visualization">

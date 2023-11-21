### Task Overview
Automated Machine Learning, or AutoML, is a process that automates the repetitive and time-consuming tasks involved in developing machine learning models. This helps data scientists, analysts, and developers to create models more efficiently and with higher quality, resulting in increased productivity and scalability.

AutoML Object Detection enables you to train machine learning models to detect and locate objects of interest in an image. It is a computer vision task that involves identifying the position and boundaries of objects in an image, and classifying the objects into different categories.

With this functionality, you can:
* Directly use datasets coming from [Azure Machine Learning data labeling](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-image-labeling-projects?view=azureml-api-2)
* Utilize labeled data to create image models without any training code.
* Enhance model performance by selecting the appropriate algorithm and fine-tuning the hyperparameters selecting the appropriate algorithm from a [large selection of models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=cli#supported-model-architectures) or let AutoML find the best model for you.
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

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image object detection|Image object detection|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip)|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/automl-image-object-detection-task-fridge-items.ipynb" target="_blank">[fridgeobjects-object-detection.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/automl-image-object-detection-task-fridge-items.ipynb)</a>|<a href="https://github.com/Azure/azureml-examples/tree/sdk-preview/cli/jobs/automl-standalone-jobs/cli-automl-image-object-detection-task-fridge-items" target="_blank">cli-automl-image-object-detection-task-fridge-items.yml</a>


### Sample inputs and outputs

#### Sample input

```json
{
  "input_data": {
    "columns": [
      "image"
    ],
    "index": [0, 1],
    "data": ["image1", "image2"]
  }
}
```

Note: "image1" and "image2" string should be in base64 format or publicly accessible urls.


#### Sample output

```json
[
    {
        "boxes": [
            {
                "box": {
                    "topX": 0.1,
                    "topY": 0.2,
                    "bottomX": 0.8,
                    "bottomY": 0.7
                },
                "label": "carton",
                "score": 0.98
            }
        ]
    },
    {
        "boxes": [
            {
                "box": {
                    "topX": 0.2,
                    "topY": 0.3,
                    "bottomX": 0.6,
                    "bottomY": 0.5
                },
                "label": "can",
                "score": 0.97
            }
        ]
    }
]
```

Note: Please refer to object detection output <a href="https://learn.microsoft.com/en-us/azure/machine-learning/reference-automl-images-schema?view=azureml-api-2#object-detection-1" target="_blank">data schema</a> for more detail.

#### Model inference - visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/yolov5_tao_vis.png" alt="od visualization">

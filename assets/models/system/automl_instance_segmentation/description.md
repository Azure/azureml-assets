Automated Machine Learning, or AutoML, is a process that automates the repetitive and time-consuming tasks involved in developing machine learning models. This helps data scientists, analysts, and developers to create models more efficiently and with higher quality, resulting in increased productivity and scalability.
AutoML Image Instance Segmentation enables you to train machine learning models to identify and separate individual objects within an image, including detecting the boundaries of each object and assigning a unique label to each object. The goal of instance segmentation is to produce a pixel-wise segmentation map of the image, where each pixel is assigned to a specific object instance.

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

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image Instance Segmentation|Image instance segmentation|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjectsMask.zip)|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-image-instance-segmentation-task-fridge-items/automl-image-instance-segmentation-task-fridge-items.ipynb" target="_blank">[automl-image-instance-segmentation-task-fridge-items.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/automl-image-object-detection-task-fridge-items.ipynb)</a>|<a href="https://github.com/Azure/azureml-examples/tree/sdk-preview/cli/jobs/automl-standalone-jobs/cli-automl-image-instance-segmentation-task-fridge-items" target="_blank">cli-automl-image-instance-segmentation-task-fridge-items.yml</a>


### Sample inputs and outputs (for real-time inference)

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
Note:
- "image1" and "image2" should be strings in `base64` format.


#### Sample output

```json
[
    {
       "boxes": [
          {
             "box": {
                "topX": 0.679,
                "topY": 0.491,
                "bottomX": 0.926,
                "bottomY": 0.810
             },
             "label": "can",
             "score": 0.992,
             "polygon": [
                [
                   0.82, 0.811, 0.771, 0.810, 0.758, 0.805, 0.741, 0.797, 0.735, 0.791, 0.718, 0.785, 0.715, 0.778, 0.706, 0.775, 0.696, 0.758, 0.695, 0.717, 0.698, 0.567, 0.705, 0.552, 0.706, 0.540, 0.725, 0.520, 0.735, 0.505, 0.745, 0.502, 0.755, 0.493
                ]
             ]
          },
          {
             "box": {
                "topX": 0.220,
                "topY": 0.298,
                "bottomX": 0.397,
                "bottomY": 0.601
             },
             "label": "milk_bottle",
             "score": 0.989,
             "polygon": [
                [
                   0.365, 0.602, 0.273, 0.602, 0.26, 0.595, 0.263, 0.588, 0.251, 0.546, 0.248, 0.501, 0.25, 0.485, 0.246, 0.478, 0.245, 0.463, 0.233, 0.442, 0.231, 0.43, 0.226, 0.423, 0.226, 0.408, 0.234, 0.385, 0.241, 0.371, 0.238, 0.345, 0.234, 0.335, 0.233, 0.325, 0.24, 0.305, 0.586, 0.38, 0.592, 0.375, 0.598, 0.365
                ]
             ]
          },
          {
             "box": {
                "topX": 0.433,
                "topY": 0.280,
                "bottomX": 0.621,
                "bottomY": 0.679
             },
             "label": "water_bottle",
             "score": 0.988,
             "polygon": [
                [
                   0.576, 0.680, 0.501, 0.680, 0.475, 0.675, 0.460, 0.625, 0.445, 0.630, 0.443, 0.572, 0.440, 0.560, 0.435, 0.515, 0.431, 0.501, 0.431, 0.433, 0.433, 0.426, 0.445, 0.417, 0.456, 0.407, 0.465, 0.381, 0.468, 0.327, 0.471, 0.318
                ]
             ]
          }
       ]
    }
]
```

Note: Please refer to instance segmentation output <a href="https://learn.microsoft.com/azure/machine-learning/reference-automl-images-schema?view=azureml-api-2#instance-segmentation-1" target="_blank">data schema</a> for more detail.

#### Model inference - visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/markrcnn_resnet50_fpn_tao_vis.png" alt="is visualization">
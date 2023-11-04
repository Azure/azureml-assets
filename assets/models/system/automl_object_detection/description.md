### Task Overview
<b>Image Object Detection</B> is a computer vision task in which the goal is to detect and locate objects of interest in an image. The task involves identifying the position and boundaries of objects in an image, and classifying the objects into different categories. 

<b>Automated ML</b> supports model training for computer vision tasks like  object detection. Authoring AutoML models for computer vision tasks is currently supported via the Azure Machine Learning Python SDK. The resulting experimentation trials, models, and outputs are accessible from the Azure Machine Learning studio UI.

Please see <a href="https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=cli#supported-model-algorithms" target="_blank">documentation</a> to get more information.

### Model Zoo
Model Architecture|Model Name(s)|Reference
|--|--|--|
YOLOv5|`yolov5`|<a href="https://github.com/ultralytics/yolov5" target="_blank">[1]</a>
Faster RCNN ResNet FPN|`fasterrcnn_resnet18_fpn`<br>`fasterrcnn_resnet34_fpn`<br>`fasterrcnn_resnet50_fpn`<br>`fasterrcnn_resnet101_fpn`<br>`fasterrcnn_resnet152_fpn`|<a href="https://arxiv.org/abs/1612.03144" target="_blank">[2]</a>
RetinaNet ResNet FPN|`retinanet_resnet50_fpn`|<a href="https://arxiv.org/abs/1708.02002" target="_blank">[3]</a>



### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Batch Scoring|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items-batch-scoring/image-object-detection-batch-scoring-non-mlflow-model.ipynb" target="_blank">image-object-detection-batch-scoring-non-mlflow-model.ipynb</a>|


### Finetune samples

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

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_yolof_r50_c5_8x8_1x_coco_OD.png" alt="od visualization">

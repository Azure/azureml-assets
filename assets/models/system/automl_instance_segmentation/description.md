### Task Overview
<b>Image Instance Segmentation</B> is a computer vision task that involves identifying and separating individual objects within an image, including detecting the boundaries of each object and assigning a unique label to each object. The goal of instance segmentation is to produce a pixel-wise segmentation map of the image, where each pixel is assigned to a specific object instance.

<b>Automated ML</b> supports model training for computer vision tasks like instance segmentation. Authoring AutoML models for computer vision tasks is currently supported via the Azure Machine Learning Python SDK. The resulting experimentation trials, models, and outputs are accessible from the Azure Machine Learning studio UI.

Please see <a href="https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2&tabs=cli#supported-model-algorithms" target="_blank">documentation</a> to get more information.

### Model Zoo
Model Architecture|Model Name(s)|Reference
|--|--|--|
MaskRCNN ResNet FPN	|`maskrcnn_resnet18_fpn`<br>`maskrcnn_resnet34_fpn`<br>`maskrcnn_resnet50_fpn`<br>`maskrcnn_resnet101_fpn`<br>`maskrcnn_resnet152_fpn`|<a href="https://arxiv.org/abs/1703.06870" target="_blank">[1]</a>


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

Note: "image1" and "image2" string should be in base64 format or publicly accessible urls.


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

Note: Please refer to instance segmentation output <a href="https://learn.microsoft.com/en-us/azure/machine-learning/reference-automl-images-schema?view=azureml-api-2#instance-segmentation-1" target="_blank">data schema</a> for more detail.

#### Model inference - visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_mask_rcnn_swin-t-p4-w7_fpn_1x_coco_IS.png" alt="is visualization">
`yolof_r50_c5_8x8_1x_coco` model is from <a href="https://github.com/open-mmlab/mmdetection/tree/v2.28.2" target="_blank">OpenMMLab's MMDetection library</a>. This paper revisits feature pyramids networks (FPN) for one-stage detectors and points out that the success of FPN is due to its divide-and-conquer solution to the optimization problem in object detection rather than multi-scale feature fusion. From the perspective of optimization, we introduce an alternative way to address the problem instead of adopting the complex feature pyramids - {\em utilizing only one-level feature for detection}. Based on the simple and efficient solution, we present You Only Look One-level Feature (YOLOF). In our method, two key components, Dilated Encoder and Uniform Matching, are proposed and bring considerable improvements. Extensive experiments on the COCO benchmark prove the effectiveness of the proposed model. Our YOLOF achieves comparable results with its feature pyramids counterpart RetinaNet while being 2.5× faster. Without transformer layers, YOLOF can match the performance of DETR in a single-level feature manner with 7× less training epochs. With an image size of 608×608, YOLOF achieves 44.3 mAP running at 60 fps on 2080Ti, which is 13% faster than YOLOv4.

# Training Details

## Training Data

The model developers used COCO dataset for training the model.

## Training Procedure

Training Techniques:

- SGD with Momentum
- Weight Decay

Training Resources: 8x V100 GPUs

Training Memory (GB): 8.3

Epochs: 12

# Evaluation Results

box AP: 37.5

# License

apache-2.0

# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-object-detection" target="_blank">image-object-detection-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-object-detection" target="_blank">image-object-detection-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-image-object-detection" target="_blank">image-object-detection-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-image-object-detection" target="_blank">image-object-detection-batch-endpoint.sh</a>

# Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image object detection|Image object detection|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip)|<a href="https://aka.ms/azureml-ft-sdk-image-object-detection" target="_blank">fridgeobjects-object-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-image-object-detection" target="_blank">fridgeobjects-object-detection.sh</a>

# Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
Image object detection|Image object detection|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip)|<a href="https://aka.ms/azureml-evaluation-sdk-image-object-detection" target="_blank">image-object-detection.ipynb</a>|

# Sample input and output

### Sample input

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

### Sample output

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

#### Visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_yolof_r50_c5_8x8_1x_coco_OD.png" alt="od visualization">

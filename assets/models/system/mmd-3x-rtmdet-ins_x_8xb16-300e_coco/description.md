In this paper, we aim to design an efficient real-time object detector that exceeds the YOLO series and is easily extensible for many object recognition tasks such as instance segmentation and rotated object detection. To obtain a more efficient model architecture, we explore an architecture that has compatible capacities in the backbone and neck, constructed by a basic building block that consists of large-kernel depth-wise convolutions. We further introduce soft labels when calculating matching costs in the dynamic label assignment to improve accuracy. Together with better training techniques, the resulting object detector, named RTMDet, achieves 52.8% AP on COCO with 300+ FPS on an NVIDIA 3090 GPU, outperforming the current mainstream industrial detectors. RTMDet achieves the best parameter-accuracy trade-off with tiny/small/medium/large/extra-large model sizes for various application scenarios, and obtains new state-of-the-art performance on real-time instance segmentation and rotated object detection. We hope the experimental results can provide new insights into designing versatile real-time object detectors for many object recognition tasks.

> The above abstract is from mmdetection website. Review the <a href="https://github.com/open-mmlab/mmdetection/tree/v3.1.0/configs/rtmdet" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-instance-segmentation" target="_blank">image-instance-segmentation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-instance-segmentation" target="_blank">image-instance-segmentation-online-endpoint.sh</a>
Batch|<a href="https://aka.ms/azureml-infer-batch-sdk-image-instance-segmentation" target="_blank">image-instance-segmentation-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-image-instance-segmentation" target="_blank">image-instance-segmentation-batch-endpoint.sh</a>

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image instance segmentation|Image instance segmentation|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjectsMask.zip)|<a href="https://aka.ms/azureml-ft-sdk-image-instance-segmentation" target="_blank">fridgeobjects-instance-segmentation.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-image-instance-segmentation" target="_blank">fridgeobjects-instance-segmentation.sh</a>

### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Image instance segmentation|Image instance segmentation|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjectsMask.zip)|<a href="https://aka.ms/azureml-evaluation-sdk-image-instance-segmentation" target="_blank">image-instance-segmentation.ipynb</a>|

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
                    "topX": 0.1,
                    "topY": 0.2,
                    "bottomX": 0.8,
                    "bottomY": 0.7
                },
                "label": "carton",
                "score": 0.98,
                "polygon": [
                    [ 0.576, 0.680,  …]
                ]
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
                "score": 0.97,
                "polygon": [
                    [ 0.58, 0.7,  …]
                ]
            }
        ]
    }
]
```

Note: Please refer to instance segmentation output <a href="https://learn.microsoft.com/en-us/azure/machine-learning/reference-automl-images-schema?view=azureml-api-2#instance-segmentation-1" target="_blank">data schema</a> for more detail.

#### Model inference - visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_mmd-3x-rtmdet-ins_x_8xb16-300e_coco.png" alt="is visualization">


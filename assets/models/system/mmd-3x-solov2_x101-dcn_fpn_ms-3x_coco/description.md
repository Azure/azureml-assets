In this work, we aim at building a simple, direct, and fast instance segmentation framework with strong performance. We follow the principle of the SOLO method of Wang et al. "SOLO: segmenting objects by locations". Importantly, we take one step further by dynamically learning the mask head of the object segmenter such that the mask head is conditioned on the location. Specifically, the mask branch is decoupled into a mask kernel branch and mask feature branch, which are responsible for learning the convolution kernel and the convolved features respectively. Moreover, we propose Matrix NMS (non maximum suppression) to significantly reduce the inference time overhead due to NMS of masks. Our Matrix NMS performs NMS with parallel matrix operations in one shot, and yields better results. We demonstrate a simple direct instance segmentation system, outperforming a few state-of-the-art methods in both speed and accuracy. A light-weight version of SOLOv2 executes at 31.3 FPS and yields 37.1% AP. Moreover, our state-of-the-art results in object detection (from our mask byproduct) and panoptic segmentation show the potential to serve as a new strong baseline for many instance-level recognition tasks besides instance segmentation.

> The above abstract is from mmdetection website. Review the <a href="https://github.com/open-mmlab/mmdetection/tree/v3.1.0/configs/solov2" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_mask_rcnn_swin-t-p4-w7_fpn_1x_coco_IS.png" alt="is visualization">

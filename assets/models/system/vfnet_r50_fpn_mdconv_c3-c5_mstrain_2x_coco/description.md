`vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco` model is from <a href="https://github.com/open-mmlab/mmdetection/tree/v2.28.2" target="_blank">OpenMMLab's MMDetection library</a>. This model is reported to obtain <a href="https://github.com/open-mmlab/mmdetection/blob/e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1/configs/vfnet/metafile.yml#L55" target="_blank">box AP of 48.0 for object-detection task on COCO dataset</a>. To understand the naming style used, please refer to <a href="https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/config.html#config-name-style" target="_blank">MMDetection's Config Name Style</a>.

Accurately ranking the vast number of candidate detections is crucial for dense object detectors to achieve high performance. Prior work uses the classification score or a combination of classification and predicted localization scores to rank candidates. However, neither option results in a reliable ranking, thus degrading detection performance. In this paper, we propose to learn an Iou-aware Classification Score (IACS) as a joint representation of object presence confidence and localization accuracy. We show that dense object detectors can achieve a more accurate ranking of candidate detections based on the IACS. We design a new loss function, named Varifocal Loss, to train a dense object detector to predict the IACS, and propose a new star-shaped bounding box feature representation for IACS prediction and bounding box refinement. Combining these two new components and a bounding box refinement branch, we build an IoU-aware dense object detector based on the FCOS+ATSS architecture, that we call VarifocalNet or VFNet for short. Extensive experiments on MS COCO show that our VFNet consistently surpasses the strong baseline by ∼2.0 AP with different backbones. Our best model VFNet-X-1200 with Res2Net-101-DCN achieves a single-model single-scale AP of 55.1 on COCO test-dev, which is state-of-the-art among various object detectors.

> The above abstract is from MMDetection website. Review the <a href="https://github.com/open-mmlab/mmdetection/tree/v2.28.2/configs/vfnet" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

> <b>Deprecation Warning</b>: This model is only compatible with mmdet <= 2.28 and is deprecated. We recommend using `mmd-3x-vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco`, the corresponding mmdet >= 3.1.0 model instead, from <a href="https://ml.azure.com/model/catalog" target="_blank">the AzureML model catalog</a>. In our model catalog, the models prefixed with mmdet-3x are compatible with mmdet >= 3.1.0.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-object-detection" target="_blank">image-object-detection-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-object-detection" target="_blank">image-object-detection-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-image-object-detection" target="_blank">image-object-detection-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-image-object-detection" target="_blank">image-object-detection-batch-endpoint.sh</a>

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Image object detection|Image object detection|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip)|<a href="https://aka.ms/azureml-ft-sdk-image-object-detection" target="_blank">fridgeobjects-object-detection.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-image-object-detection" target="_blank">fridgeobjects-object-detection.sh</a>

### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
Image object detection|Image object detection|[fridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip)|<a href="https://aka.ms/azureml-evaluation-sdk-image-object-detection" target="_blank">image-object-detection.ipynb</a>|

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

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_OD.png" alt="od visualization">

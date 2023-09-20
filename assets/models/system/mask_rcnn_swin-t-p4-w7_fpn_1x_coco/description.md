This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures.

> The above abstract is from mmdetection website. Review the <a href="https://github.com/open-mmlab/mmdetection/tree/v2.28.2/configs/swin" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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

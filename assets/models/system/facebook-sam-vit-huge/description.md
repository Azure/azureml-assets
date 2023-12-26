The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

The SAM model is made up of 3 modules:

- The `VisionEncoder`: a VIT based image encoder. It computes the image embeddings using attention on patches of the image. Relative Positional Embedding is used.
- The `PromptEncoder`: generates embeddings for points and bounding boxes
- The `MaskDecoder`: a two-ways transformer which performs cross attention between the image embedding and the point embeddings (->) and between the point embeddings and the image embeddings. The outputs are fed
- The `Neck`: predicts the output masks based on the contextualized masks produced by the `MaskDecoder`.

# Training Details

## Training Data

See [here](https://ai.facebook.com/datasets/segment-anything/) for an overview of the datastet.

# License

apache-2.0

# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-mask-generation" target="_blank">mask-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-mask-generation" target="_blank">mask-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-mask-generation" target="_blank">mask-generation-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-mask-generation" target="_blank">mask-generation-batch-endpoint.sh</a>

# Sample input and output

### Sample input

```json
{
  "input_data": {
    "columns": [
      "image",
      "input_points",
      "input_boxes",
      "input_labels",
      "multimask_output"
    ],
    "index": [0],
    "data": [["image1", "", "[[650, 900, 1000, 1250]]", "", false]]
  },
  "params": {}
}
```

Note: "image1" string should be in base64 format or publicly accessible urls.


### Sample output

```json
[
    {
        "predictions": [
          0: {
            "mask_per_prediction": [
              0: {
                "encoded_binary_mask": "encoded_binary_mask1",
                "iou_score": 0.85
              }
            ]
          }
        ]
    },
]
```

Note: "encoded_binary_mask1" string is in base64 format.

#### Visualization of inference result for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_facebook-sam-vit-huge.png" alt="mask generation visualization">

The Segment Anything Model (SAM) is an innovative image segmentation tool capable of creating high-quality object masks from simple input prompts. Trained on a massive dataset comprising 11 million images and 1.1 billion masks, SAM demonstrates strong zero-shot capabilities, effectively adapting to new image segmentation tasks without prior specific training. The model's impressive performance matches or exceeds prior models that operated under full supervision.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/facebook/sam-vit-large" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-mask-generation" target="_blank">mask-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-mask-generation" target="_blank">mask-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-mask-generation" target="_blank">mask-generation-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-mask-generation" target="_blank">mask-generation-batch-endpoint.sh</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

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


#### Sample output

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

Note: "encoded_binary_mask1" string should be in base64 format.


#### Model inference - visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_facebook-sam-vit-base.png" alt="mask generation visualization">

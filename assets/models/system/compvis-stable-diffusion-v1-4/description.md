This is a model that can be used to generate and modify images based on text prompts. It is a Latent Diffusion Model that uses a fixed, pretrained text encoder (CLIP ViT-L/14) as suggested in the Imagen paper.
> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-text-to-image" target="_blank">text-to-image-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-text-to-image" target="_blank">text-to-image-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-to-image" target="_blank">text-to-image-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-text-to-image" target="_blank">text-to-image-batch-endpoint.sh</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
  "input_data": {
    "columns": [
      "text_prompts"
    ],
    "index": [0, 1],
    "data": ["a photograph of a dog in a bus", "a photograph of an astromer in a bus"]
  }
}
```

#### Sample output

```json
[
    {
        "generated_images": ["image1", "image2"],
        "nsfw_content_detected": ["True", "False"]
    }
]

Note: "image1" and "image2" string are base64 format.
```

#### Model inference - visualization for a sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_microsoft-swinv2-base-patch4-window12-192-22k_MC.png" alt="mc visualization">

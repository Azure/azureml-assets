___runwayml/stable-diffusion-v1-5___ is a powerful text-to-image latent diffusion model capable of generating photo-realistic images given any text input. The model uses a fixed pretrained text encoder (CLIP ViT-L/14) as suggested in the <a href="https://arxiv.org/abs/2205.11487" target="_blank">Imagen paper</a>, for generating realistic images from text input. Stable-Diffusion-v1-5 model was fine-tuned from an  earlier version, stable-diffusion-v1-2, on laion-aesthetics v2.5+ dataset. The model's training process involves encoding images and text prompts, and it uses a reconstruction objective. The model has various applications in research, art, education, and creative tools. However, there are strict guidelines for the model's use to prevent misuse and malicious activities. It should not be used to create harmful, offensive, or discriminatory content. Additionally, the model has limitations, such as difficulties with photorealism, rendering legible text, and generating complex compositions. The model's training data includes the LAION-2B dataset, primarily containing English descriptions, which can lead to biases and limitations in generating non-English content. To enhance safety, a Safety Checker is recommended for use with this model.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-text-to-image" target="_blank">text-to-image-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-text-to-image" target="_blank">text-to-image-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-to-image" target="_blank">text-to-image-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-text-to-image" target="_blank">text-to-image-batch-endpoint.sh</a>

<h3> Inference with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart", target="_blank">Azure AI Content Safety (AACS)</a> samples </h3>

Inference type|Python sample (Notebook)
|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image" target="_blank">safe-text-to-image-online-deployment.ipynb</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image" target="_blank">safe-text-to-image-batch-endpoint.ipynb</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
   "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse", "lion holding hunted deer in grass fields"],
        "index": [0, 1]
    }
}
```

#### Sample output

```json
[
    {
        "prompt": "a photograph of an astronaut riding a horse",
        "generated_image": "image1",
        "nsfw_content_detected": False
    },
    {
        "prompt": "lion holding hunted deer in grass fields",
        "generated_image": "image2",
        "nsfw_content_detected": True
    }
]
```

> Note:
>
> - "image1" and "image2" strings are base64 format.
> - If "nsfw_content_detected" is True then generated image will be totally black.

#### Model inference: visualization for the prompt - "a photograph of an astronaut riding a horse"

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_runwayml_stable_diffusion_v1_5.jpg" alt="runwayml_stable_diffusion_v1_5 visualization">

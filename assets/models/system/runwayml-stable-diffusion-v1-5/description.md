`runwayml/stable-diffusion-v1-5` is a powerful text-to-image latent diffusion model capable of generating photo-realistic images given any text input. The model uses a fixed pretrained text encoder (CLIP ViT-L/14) as suggested in the <a href="https://arxiv.org/abs/2205.11487" target="_blank">Imagen paper</a>, for generating realistic images from text input. Stable-Diffusion-v1-5 model was fine-tuned from an  earlier version, stable-diffusion-v1-2, on laion-aesthetics v2.5+ dataset. The model's training process involves encoding images and text prompts, and it uses a reconstruction objective. The model has various applications in research, art, education, and creative tools. However, there are strict guidelines for the model's use to prevent misuse and malicious activities. It should not be used to create harmful, offensive, or discriminatory content. Additionally, the model has limitations, such as difficulties with photorealism, rendering legible text, and generating complex compositions. The model's training data includes the LAION-2B dataset, primarily containing English descriptions, which can lead to biases and limitations in generating non-English content. To enhance safety, a Safety Checker is recommended for use with this model.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-text-to-image" target="_blank">text-to-image-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-text-to-image" target="_blank">text-to-image-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-to-image" target="_blank">text-to-image-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-text-to-image" target="_blank">text-to-image-batch-endpoint.sh</a>

### Inference with Azure AI Content Safety (AACS) samples

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
        "data": ["a photograph of an astronaut riding a horse", "a photograph of a cat riding a horse"],
        "index": [0]
    }
}
```

#### Sample output

```json
[
    {
        "generated_image": ["image1", "image2"],
        "nsfw_content_detected": ["True", "False"]
    }
]

Note: "image1" and "image2" string are base64 format.
```

#### Model inference - visualization for a sample prompt

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_runwayml_stable_diffusion_v1_5.png" alt="runwayml_stable_diffusion_v1_5 visualization">

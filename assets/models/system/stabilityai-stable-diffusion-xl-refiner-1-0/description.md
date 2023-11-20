___stabilityai/stable-diffusion-xl-refiner-1.0___ employs an ensemble of expert modules in a pipeline for latent diffusion. The process involves using a base model to generate noisy latents, which are then refined using a specialized denoising model. The base model can function independently. Alternatively, a two-stage pipeline involves generating latents with the base model and then refining them using a high-resolution model and the SDEdit technique. The second approach is slightly slower due to more function evaluations.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-text-to-image-generation" target="_blank">image-text-to-image-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-text-to-image-generation" target="_blank">image-text-to-image-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-image-text-to-image-generation" target="_blank">image-text-to-image-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-image-text-to-image-generation" target="_blank">image-text-to-image-batch-endpoint.sh</a>

<h3> Inference with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart", target="_blank">Azure AI Content Safety (AACS)</a> samples </h3>

Inference type|Python sample (Notebook)
|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-safe-image-text-to-image-generation" target="_blank">safe-image-text-to-image-online-endpoint.ipynb</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-safe-image-text-to-image-generation" target="_blank">safe-image-text-to-image-batch-endpoint.ipynb</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
   "input_data": {
        "columns": ["prompt", "image"],
        "data": [
            {
                "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
                "image": "image1",
            },
            {
                "prompt": "Face of a green cat, high resolution, sitting on a park bench",
                "image": "image2",
            }
        ],
        "index": [0, 1]
    }
}
```

> Note:
>
> - "image1" and "image2" strings are base64 format.

#### Sample output

```json
[
    {
        "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
        "generated_image": "generated_image1",
        "nsfw_content_detected": null
    },
    {
        "prompt": "Face of a green cat, high resolution, sitting on a park bench",
        "generated_image": "generated_image2",
        "nsfw_content_detected": null
    }
]
```

> Note:
>
> - "generated_image1" and "generated_image2" strings are in base64 format.
> - The `stabilityai-stable-diffusion-xl-refiner-1-0` model doesn't check for the NSFW content in generated image. We highly recommend to use the model with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart" target="_blank">Azure AI Content Safety (AACS)</a>. Please refer sample <a href="https://aka.ms/azureml-infer-sdk-safe-image-text-to-image-generation" target="_blank">online</a> and <a href="https://aka.ms/azureml-infer-batch-sdk-safe-image-text-to-image-generation" target="_blank">batch</a> notebooks for AACS integrated deployments.

#### Model inference: visualization for the prompt - "gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_gridviz_stabilityai-stable-diffusion-xl-refiner-1-0.png" alt="stabilityai-stable-diffusion-xl-refiner-1-0 input image and output visualization">

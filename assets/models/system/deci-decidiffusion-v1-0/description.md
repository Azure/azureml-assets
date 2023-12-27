`DeciDiffusion` 1.0 is an 820 million parameter latent diffusion model designed for text-to-image conversion. Trained initially on the LAION-v2 dataset and fine-tuned on the LAION-ART dataset, the model's training involved advanced techniques to improve speed, training performance, and achieve superior inference quality.

For more details, review the <a href='https://deci.ai/blog/decidiffusion-1-0-3x-faster-than-stable-diffusion-same-quality/' target='_blank'>blog</a>.

# License

creativeml-openrail++-m

# Inference Samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-text-to-image" target="_blank">text-to-image-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-text-to-image" target="_blank">text-to-image-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-to-image" target="_blank">text-to-image-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-text-to-image" target="_blank">text-to-image-batch-endpoint.sh</a>

<h1> Inference with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart", target="_blank">Azure AI Content Safety (AACS)</a> Samples </h1>

Inference type|Python sample (Notebook)
|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image" target="_blank">safe-text-to-image-online-deployment.ipynb</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image" target="_blank">safe-text-to-image-batch-endpoint.ipynb</a>

# Sample input and output

### Sample input

```json
{
   "input_data": {
        "columns": ["prompt"],
        "data": ["A photo of an astronaut riding a horse on Mars"],
        "index": [0]
    }
}
```

### Sample output

```json
[
    {
        "prompt": "A photo of an astronaut riding a horse on Mars",
        "generated_image": "image",
        "nsfw_content_detected": null
    }
]
```

> Note:
>
> - "image" string is in base64 format.
> - The `deci-decidiffusion-v1-0` model checks for the NSFW content in generated image. We highly recommend to use the model with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart" target="_blank">Azure AI Content Safety (AACS)</a>. Please refer sample <a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image" target="_blank">online</a>  and <a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image" target="_blank">batch</a> notebooks for AACS integrated deployments.

#### Visualization of inference result for a sample prompt - "a photograph of an astronaut riding a horse"

<img src='https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_deci_decidiffusion_v1_0.png' alt='deci_decidiffusion_v1_0 visualization'>
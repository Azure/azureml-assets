`DeciDiffusion` 1.0 is an 820 million parameter latent diffusion model designed for text-to-image conversion. Trained initially on the LAION-v2 dataset and fine-tuned on the LAION-ART dataset, the model's training involved advanced techniques to improve speed, training performance, and achieve superior inference quality.

DeciDiffusion 1.0 retains key elements from Stable Diffusion, like the Variational Autoencoder (VAE) and CLIP's pre-trained Text Encoder, while introducing notable improvements. But U-Net is replaced with the more efficient U-Net-NAS which is developed by Deci. This novel component streamlines the model by reducing parameters, resulting in enhanced computational efficiency.

For more details, review the <a href='https://deci.ai/blog/decidiffusion-1-0-3x-faster-than-stable-diffusion-same-quality/' target='_blank'>blog</a>.

# Training Details

## Training Procedure

This model was trained in 4 phases.

1. It was trained from scratch for 1.28 million steps at a resolution of 256x256 using 320 million samples from LAION-v2.

2. The model was trained for 870k steps at a higher resolution of 512x512 on the same dataset to capture more fine-detailed information.

3. Training for 65k steps with EMA, a different learning rate scheduler, and more qualitative data.

4. Then the model underwent fine-tuning on a 2 million sample subset of the LAION-ART dataset.

In phase 1, 8 X 8 X A100 GPUs, AdamW optimizer had been used with batch size 8192 and learning rate 1e-4. In phases 2-4, 8 X 8 X H100 GPUs, LAMB optimizer had been used with batch size 6144 and learning rate 5e-3.

# Limitations and Biases

## Limitations

The model has limitations and may not perform optimally in various scenarios. It doesn't generate entirely photorealistic images. Rendering legible text is beyond its capability. The generation of faces and human figures may lack precision. The model is primarily optimized for English captions and may not be as effective with other languages. The auto-encoding component of the model is lossy.

## Biases

DeciDiffusion primarily underwent training on subsets of LAION-v2, with a focus on English descriptions. As a result, there might be underrepresentation of non-English communities and cultures, potentially introducing bias towards white and western norms. The accuracy of outputs from non-English prompts is notably less accurate. Considering these biases, users are advised to exercise caution when using DeciDiffusion, irrespective of the input provided.

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
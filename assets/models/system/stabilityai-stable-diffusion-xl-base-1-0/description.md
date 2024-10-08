[SDXL](https://arxiv.org/abs/2307.01952) consists of an [ensemble of experts](https://arxiv.org/abs/2211.01324) pipeline for latent diffusion: 
In a first step, the base model (available here: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) is used to generate (noisy) latents, which are then further processed with a refinement model specialized for the final denoising steps. Note that the base model can be used as a standalone module.

Alternatively, we can use a two-stage pipeline as follows:
First, the base model is used to generate latents of the desired output size. In the second step, we use a specialized high-resolution model and apply a technique called SDEdit (https://arxiv.org/abs/2108.01073, also known as "img2img") to the latents generated in the first step, using the same prompt. This technique is slightly slower than the first one, as it requires more function evaluations.

The model is intended for research purposes only. Possible research areas and tasks include

- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.

# Evaluation Results

This <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/comparison.png" target="_blank">chart</a> evaluates user preference for SDXL (with and without refinement) over SDXL 0.9 and Stable Diffusion 1.5 and 2.1. The SDXL base model performs significantly better than the previous variants, and the model combined with the refinement module achieves the best overall performance.

# Limitations and Biases

## Limitations

- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.

## Bias

While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.

## Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

# License

CreativeML Open RAIL++-M License

# Inference Samples

> Note: The inferencing script of this model is optimized for high-throughput, low latency using <a href="https://github.com/microsoft/DeepSpeed-MII" target="_blank">Deepspedd-mii</a> library. Please use `version 4` of this model for inferencing using default (FP32) diffusion pipeline implementation.

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-text-to-image" target="_blank">text-to-image-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-text-to-image" target="_blank">text-to-image-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-to-image" target="_blank">text-to-image-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-text-to-image" target="_blank">text-to-image-batch-endpoint.sh</a>

<h3> Inference with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart", target="_blank">Azure AI Content Safety (AACS)</a> samples </h3>

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
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0]
    }
}
```

### Sample output

```json
[
    {
        "prompt": "a photograph of an astronaut riding a horse",
        "generated_image": "image",
        "nsfw_content_detected": null
    }
]
```

> Note:
>
> - "image" string is in base64 format.
> - The `stabilityai-stable-diffusion-xl-base-1-0` model doesn't check for the NSFW content in generated image. We highly recommend to use the model with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart" target="_blank">Azure AI Content Safety (AACS)</a>. Please refer sample <a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image" target="_blank">online</a>  and <a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image" target="_blank">batch</a> notebooks for AACS integrated deployments.

#### Visualization for the prompt - "a photograph of an astronaut riding a horse"

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_stabilityai_stable_diffusion_xl_base-1-0.png" alt="stabilityai_stable_diffusion_xl_base-1-0 visualization">

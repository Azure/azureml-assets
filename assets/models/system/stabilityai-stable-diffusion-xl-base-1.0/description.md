This `stabilityai-stable-diffusion-xl-base-1.0` model is fine-tuned from [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 

The SD-XL 1.0-base Model, developed by Stability AI, is a Diffusion-based text-to-image generative model. It utilizes a two-stage pipeline for latent diffusion, consisting of a base model and a refinement model. The base model generates noisy latents, which are then processed by a refinement model for final denoising. Alternatively, a two-stage pipeline involves using a specialized high-resolution model and applying the SDEdit technique to the latents generated in the first step.

The model is available on GitHub, and for research purposes, Stability AI recommends their generative-models repository. The model is licensed under the CreativeML Open RAIL++-M License. It is designed for various research areas, such as generating artworks, educational tools, and understanding the limitations and biases of generative models. However, it is not intended for factual or true representations of people or events.

The model has limitations, including not achieving perfect photorealism, struggling with tasks involving compositionality, and facing challenges with rendering faces and legible text. The autoencoding part of the model is lossy, and there is a potential for reinforcing social biases in generated content. The model's deployment is restricted to safe applications, and certain use cases, like generating factual content, are considered out-of-scope for its abilities.

You can find more examples in optimum https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models#stable-diffusion-xl

### Misuse and Malicious Use

Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

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

### Supported Parameters

- num_inference_steps: The number of de-noising steps. More de-noising steps usually lead to a higher quality image at the expense of slower inference, defaults to 50.
- guidance_scale: A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`, defaults to 7.5.

> These `parameters` are optional inputs. If you need support for new parameters, please file a support ticket.

### Sample input

```json
{
   "input_data": {
        "columns": ["prompt"],
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0],
        "parameters": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5
        }
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
> - The `stabilityai-stable-diffusion-xl-base-1.0` model doesn't check for the NSFW content in generated image. We highly recommend to use the model with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart" target="_blank">Azure AI Content Safety (AACS)</a>. Please refer sample <a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image" target="_blank">online</a>  and <a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image" target="_blank">batch</a> notebooks for AACS integrated deployments.

#### Visualization for the prompt - "a photograph of an astronaut riding a horse"

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_stabilityai_stable_diffusion_2_1.png" alt="stabilityai_stable_diffusion_2_1 visualization">

__Deci/DeciDiffusion-v1-0__ model model proficient in generating lifelike images based on textual descriptions. Grounded in latent diffusion methodology, it systematically converts random noise into an image aligning with the provided text. This innovation is the creation of Deci, an entity specializing in optimizing AI models for enhanced speed and efficiency. Boasting a threefold increase in speed compared to Stable Diffusion 1.5, DeciDiffusion 1.0 achieves this feat while maintaining image quality. The model underwent training on a substantial dataset named LAION-v2, encompassing images and texts, and underwent additional fine-tuning using a selection of artistic images known as LAION-ART1.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/Deci/DeciDiffusion-v1-0" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
        "data": ["a photograph of an astronaut riding a horse"],
        "index": [0]
    }
}
```

#### Sample output

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
> - The `stabilityai-stable-diffusion-2-1` model doesn't check for the NSFW content in generated image. We highly recommend to use the model with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart" target="_blank">Azure AI Content Safety (AACS)</a>. Please refer sample <a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image" target="_blank">online</a>  and <a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image" target="_blank">batch</a> notebooks for AACS integrated deployments.

#### Model inference: visualization for the prompt - "a photograph of an astronaut riding a horse"
Need to add
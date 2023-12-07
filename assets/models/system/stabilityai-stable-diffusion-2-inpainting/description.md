___stabilityai/stable-diffusion-2-inpainting___ model is a continuation of the stable-diffusion-2-base model, with an additional 200,000 steps of training. It utilizes a mask-generation strategy introduced in LAMA and combines this with latent Variational Autoencoder (VAE) representations of the masked image to provide additional conditioning during the training process. LAION-5B and subsets are used as training data, which is further filtered using LAION's NSFW detector, with a "p_unsafe" score of 0.1 (conservative). The model should not be used to create harmful, offensive, or discriminatory content. Additionally, the model has limitations, such as difficulties with photorealism, rendering legible text, and generating complex compositions. The model's training data, LAION-5B, primarily contains English descriptions, which can lead to biases and limitations in generating non-English content. To enhance safety, a Safety Checker is recommended for use with this model.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-text-to-image-inpainting" target="_blank">text-to-image-inpainting-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-text-to-image-inpainting" target="_blank">text-to-image-inpainting-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-to-image-inpainting" target="_blank">text-to-image-inpainting-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-text-to-image-inpainting" target="_blank">text-to-image-inpainting-batch-endpoint.sh</a>

<h3> Inference with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart", target="_blank">Azure AI Content Safety (AACS)</a> samples </h3>

Inference type|Python sample (Notebook)
|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image-inpainting" target="_blank">safe-text-to-image-inpainting-online-deployment.ipynb</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image-inpainting" target="_blank">safe-text-to-image-inpainting-batch-endpoint.ipynb</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
   "input_data": {
        "columns": ["prompt", "image", "mask"],
        "data": [
            {
                "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
                "image": "image1",
                "mask_image": "mask1"
            },
            {
                "prompt": "Face of a green cat, high resolution, sitting on a park bench",
                "image": "image2",
                "mask_image": "mask2"
            }
        ],
        "index": [0, 1]
    }
}
```

> Note:
>
> - "image1" and "image2" strings are base64 format.
> - "mask1" and "mask2" strings are base64 format.

#### Sample output

```json
[
    {
        "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
        "generated_image": "inpainted_image1",
        "nsfw_content_detected": null
    },
    {
        "prompt": "Face of a green cat, high resolution, sitting on a park bench",
        "generated_image": "inpainted_image2",
        "nsfw_content_detected": null
    }
]
```

> Note:
>
> - "inpainted_image1" and "inpainted_image2" strings are base64 format.
> - The `stabilityai-stable-diffusion-2-inpainting` model doesn't check for the NSFW content in generated image. We highly recommend to use the model with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart" target="_blank">Azure AI Content Safety (AACS)</a>. Please refer sample <a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image-inpainting" target="_blank">online</a> and <a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image-inpainting" target="_blank">batch</a> notebooks for AACS integrated deployments.

#### Model inference: visualization for the prompt - "a small flower vase featuring a blend of yellow and orange"

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_gridviz_stabilityai_stable_diffusion_2_inpainting.png" alt="stabilityai_stable_diffusion_2_inpainting input image mask image and output visualization">

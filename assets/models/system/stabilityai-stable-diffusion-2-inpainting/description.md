This `stable-diffusion-2-inpainting` model is resumed from [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) (`512-base-ema.ckpt`) and trained for another 200k steps. Follows the mask-generation strategy presented in [LAMA](https://github.com/saic-mdal/lama) which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning.

The model is intended for research purposes only. Possible research areas and tasks include

- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.

# Training Details

## Training Data

The model developers used the following dataset for training the model:

- LAION-5B and subsets (details below). The training data is further filtered using LAION's NSFW detector, with a "p_unsafe" score of 0.1 (conservative). For more details, please refer to LAION-5B's [NeurIPS 2022](https://openreview.net/forum?id=M3Y74vmsMcY) paper and reviewer discussions on the topic.

## Training Procedure

Stable Diffusion v2 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training,

- Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4
- Text prompts are encoded through the OpenCLIP-ViT/H text-encoder.
- The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.
- The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called _v-objective_, see https://arxiv.org/abs/2202.00512.
We currently provide the following checkpoint:

- `512-inpainting-ema.ckpt`: Resumed from `512-base-ema.ckpt` and trained for another 200k steps. Follows the mask-generation strategy presented in [LAMA](https://github.com/saic-mdal/lama) which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning.
The additional input channels of the U-Net which process this extra information were zero-initialized. The same strategy was used to train the [1.5-inpainting checkpoint](https://github.com/saic-mdal/lama).

- **Hardware:** 32 x 8 x A100 GPUs
- **Optimizer:** AdamW
- **Gradient Accumulations**: 1
- **Batch:** 32 x 8 x 2 x 4 = 2048
- **Learning rate:** warmup to 0.0001 for 10,000 steps and then kept constant

# Limitations and Biases

## Limitations

- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The model was trained mainly with English captions and will not work as well in other languages.
- The autoencoding part of the model is lossy
- The model was trained on a subset of the large-scale dataset
  [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content. To partially mitigate this, we have filtered the dataset using LAION's NFSW detector (see Training section).

## Bias

While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. Stable Diffusion vw was primarily trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), which consists of images that are limited to English descriptions. Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. This affects the overall output of the model, as white and western cultures are often set as the default. Further, the ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts. Stable Diffusion v2 mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent.

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Misuse and Malicious Use

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

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-text-to-image-inpainting" target="_blank">text-to-image-inpainting-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-text-to-image-inpainting" target="_blank">text-to-image-inpainting-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-to-image-inpainting" target="_blank">text-to-image-inpainting-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-text-to-image-inpainting" target="_blank">text-to-image-inpainting-batch-endpoint.sh</a>

<h3> Inference with <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/studio-quickstart", target="_blank">Azure AI Content Safety (AACS)</a> samples </h3>

Inference type|Python sample (Notebook)
|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-safe-text-to-image-inpainting" target="_blank">safe-text-to-image-inpainting-online-deployment.ipynb</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-safe-text-to-image-inpainting" target="_blank">safe-text-to-image-inpainting-batch-endpoint.ipynb</a>

# Sample input and output

### Sample input

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

### Sample output

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

#### Visualization for the prompt - "a small flower vase featuring a blend of yellow and orange"

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_gridviz_stabilityai_stable_diffusion_2_inpainting.png" alt="stabilityai_stable_diffusion_2_inpainting input image mask image and output visualization">

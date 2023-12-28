`BLIP` (Bootstrapping Language-Image Pre-training) designed for unified vision-language understanding and generation is a new VLP framework that expands the scope of downstream tasks compared to existing methods. The framework encompasses two key contributions from both model and data perspectives.

1. BLIP incorporates the Multi-modal Mixture of Encoder-Decoder (MED), an innovative model architecture designed to facilitate effective multi-task pre-training and flexible transfer learning. This model is jointly pre-trained using three vision-language objectives: image-text contrastive learning, image-text matching, and image-conditioned language modeling.

2. BLIP introduces Captioning and Filtering (CapFilt), a distinctive dataset bootstrapping method aimed at learning from noisy image-text pairs. The pre-trained MED is fine-tuned into a captioner that generates synthetic captions from web images, and a filter that removes noisy captions from both the original web texts and synthetic texts.

Authors of BLIP make following key observations based on extensive experiments and analysis. The collaboration between the captioner and filter significantly enhances performance across diverse downstream tasks through caption bootstrapping, with greater diversity in captions leading to more substantial gains. BLIP achieves  state-of-the-art performance in various vision-language tasks, including image-text retrieval, image captioning, visual question answering, visual reasoning, and visual dialog. It also achieves state-of-the-art zero-shot performance when directly applied to video-language tasks such as text-to-video retrieval and videoQA.

Researchers should carefully assess the safety and fairness of the model before deploying it in any real-world applications.

Model fine-tuned on COCO dataset with the language modeling (LM) loss to generate captions given images with base architecture (with ViT base backbone). For more details on Image Captioning with BLIP, review the section 5.2 of the <a href='https://arxiv.org/abs/2201.12086' target='_blank'>original-paper</a>.

# License

BSD 3-Clause License

# Inference Samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href='https://aka.ms/azureml-infer-online-sdk-blip-image-to-text' target='_blank'>image-to-text-online-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-online-cli-blip-image-to-text' target='_blank'>image-to-text-online-endpoint.sh</a>
Batch |<a href='https://aka.ms/azureml-infer-batch-sdk-blip-image-to-text' target='_blank'>image-to-text-batch-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-batch-cli-blip-image-to-text' target='_blank'>image-to-text-batch-endpoint.sh</a>

# Sample input and output

### Sample input

```json
{
   "input_data":{
      "columns":[
         "image"
      ],
      "index":[0, 1],
      "data":[
         ["image1"],
         ["image2"]
      ]
   }
}
```
Note:
- "image1" and "image2" should be publicly accessible urls or strings in `base64` format.

### Sample output

```json
[
   {
      "text": "a box of food sitting on top of a table"
   },
   {
      "text": "a stream in the middle of a forest"
   }
]
```

#### Visualization of inference result for a sample image

For sample image below, the output text is "a stream in the middle of a forest".

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_blip_image_captioning_base.png" alt="Salesforce-BLIP-image-captioning-base">
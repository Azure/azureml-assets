`BLIP` (Bootstrapping Language-Image Pre-training) designed for unified vision-language understanding and generation is a new VLP framework that expands the scope of downstream tasks compared to existing methods. The framework encompasses two key contributions from both model and data perspectives.

1. BLIP incorporates the Multi-modal Mixture of Encoder-Decoder (MED), an innovative model architecture designed to facilitate effective multi-task pre-training and flexible transfer learning. This model is jointly pre-trained using three vision-language objectives: image-text contrastive learning, image-text matching, and image-conditioned language modeling.

2. BLIP introduces Captioning and Filtering (CapFilt), a distinctive dataset bootstrapping method aimed at learning from noisy image-text pairs. The pre-trained MED is fine-tuned into a captioner that generates synthetic captions from web images, and a filter that removes noisy captions from both the original web texts and synthetic texts.

Authors of BLIP make following key observations based on extensive experiments and analysis. The collaboration between the captioner and filter significantly enhances performance across diverse downstream tasks through caption bootstrapping, with greater diversity in captions leading to more substantial gains. BLIP achieves  state-of-the-art performance in various vision-language tasks, including image-text retrieval, image captioning, visual question answering, visual reasoning, and visual dialog. It also achieves state-of-the-art zero-shot performance when directly applied to video-language tasks such as text-to-video retrieval and videoQA.

Researchers should carefully assess the safety and fairness of the model before deploying it in any real-world applications.

In Visual Question Answering (VQA) task, the objective is to predict an answer given an image and a question. In the fine-tuning process, the pre-trained model is restructured to encode the image-question pair into multi-modal embeddings. These embeddings are then given to answer decoder. Fine-tuning of the VQA model involves using the Language Model (LM) loss, with ground-truth answers used as the target. For more details on Image Captioning with BLIP, review the section 5.3 of the <a href='https://arxiv.org/abs/2201.12086' target='_blank'>original-paper</a>.

# License

BSD 3-Clause License

# Inference Samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href='https://aka.ms/azureml-infer-online-sdk-blip-vqa' target='_blank'>visual-question-answering-online-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-online-cli-blip-vqa' target='_blank'>visual-question-answering-online-endpoint.sh</a>
Batch |<a href='https://aka.ms/azureml-infer-batch-sdk-blip-vqa' target='_blank'>visual-question-answering-batch-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-batch-cli-blip-vqa' target='_blank'>visual-question-answering-batch-endpoint.sh</a>

# Sample input and output

### Sample input

```json
{
   "input_data":{
      "columns":[
         "image",
         "text"
      ],
      "index":[0, 1],
      "data":[
         ["image1", "What is in the picture?"],
         ["image2", "How many dogs are in the picture?"]
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
      "text": "sand"
   },
   {
      "text": "1"
   }
]
```

#### Visualization of inference result for a sample image

For sample image below and text prompt "What is in the picture?", the output text is "sand".

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_blip_vqa_base.png" alt="Salesforce-BLIP-vqa-base">
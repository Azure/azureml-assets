The `BLIP` model is a pretrained deep learning model for image captioning based on the Bootstrapping Language-Image Pre-training (BLIP) framework. It is trained on the COCO dataset and generates a descriptive caption for an input image by focusing on different parts of the image using an attention mechanism and generating a sequence of words to form the caption. The model consists of a patch embedding layer, a transformer encoder, and a language decoder. The attention mechanism helps the model generate more accurate and descriptive captions by focusing on the relevant parts of the image. Researchers should carefully assess the safety and fairness of the model before deploying it in any real-world applications.

> The above summary was generated using ChatGPT. Review the <a href='https://huggingface.co/Salesforce/blip-image-captioning-base' target='_blank'>original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href='https://aka.ms/azureml-infer-sdk-blip2-image-to-text' target='_blank'>image-to-text-online-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-cli-blip2-image-to-text' target='_blank'>image-to-text-online-endpoint.sh</a>
Batch |<a href='https://aka.ms/azureml-infer-batch-sdk-blip2-image-to-text' target='_blank'>image-to-text-batch-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-batch-cli-blip2-image-to-text' target='_blank'>image-to-text-batch-endpoint.sh</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

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

#### Sample output

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

#### Model inference: Text for the sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_blip_image_captioning_base.png" alt="Salesforce-BLIP-image-captioning-base">
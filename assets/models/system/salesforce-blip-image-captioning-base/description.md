The `BLIP` framework is a new Vision-Language Pre-training (VLP) framework that can be used for both vision-language understanding and generation tasks. BLIP effectively utilizes noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. This framework achieves state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval, image captioning, and VQA. BLIP also demonstrates strong generalization ability when directly transferred to video-language tasks in a zero-shot manner. The code, models, and datasets are available for use. Researchers should carefully assess the safety and fairness of the model before deploying it in any real-world applications.

> The above summary was generated using ChatGPT. Review the <a href='https://huggingface.co/Salesforce/blip-image-captioning-base' target='_blank'>original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href='https://aka.ms/azureml-infer-online-sdk-blip-image-to-text' target='_blank'>image-to-text-online-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-online-cli-blip-image-to-text' target='_blank'>image-to-text-online-endpoint.sh</a>
Batch |<a href='https://aka.ms/azureml-infer-batch-sdk-blip-image-to-text' target='_blank'>image-to-text-batch-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-batch-cli-blip-image-to-text' target='_blank'>image-to-text-batch-endpoint.sh</a>

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

#### Model inference - image to text
For sample image below, the output text is "a stream in the middle of a forest".

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_blip_image_captioning_base.png" alt="Salesforce-BLIP-image-captioning-base">
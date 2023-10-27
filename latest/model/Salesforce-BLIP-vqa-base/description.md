`BLIP` is a new Vision-Language Pre-training (VLP) framework that excels in both understanding-based and generation-based tasks. It effectively utilizes noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. BLIP achieves state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval, image captioning, and VQA. It also demonstrates strong generalization ability when directly transferred to video-language tasks in a zero-shot manner. Code, models, and datasets are available on the official repository.

> The above summary was generated using ChatGPT. Review the <a href='https://huggingface.co/Salesforce/blip-vqa-base' target='_blank'>original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href='https://aka.ms/azureml-infer-online-sdk-blip-vqa' target='_blank'>visual-question-answering-online-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-online-cli-blip-vqa' target='_blank'>visual-question-answering-online-endpoint.sh</a>
Batch |<a href='https://aka.ms/azureml-infer-batch-sdk-blip-vqa' target='_blank'>visual-question-answering-batch-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-batch-cli-blip-vqa' target='_blank'>visual-question-answering-batch-endpoint.sh</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

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

#### Sample output

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

#### Model inference: Text for the sample image

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_blip_vqa_base.png" alt="Salesforce-BLIP-vqa-base">
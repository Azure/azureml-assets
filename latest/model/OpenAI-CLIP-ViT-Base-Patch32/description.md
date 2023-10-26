The `CLIP` model was developed by OpenAI researchers to learn about what contributes to robustness in computer vision tasks and to test the ability of models to generalize to arbitrary image classification tasks in a zero-shot manner. The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. The model was trained on publicly available image-caption data, which was gathered in a mostly non-interventionist manner. The model is intended as a research output for research communities, and the primary intended users of these models are AI researchers. The model has been evaluated on a wide range of benchmarks across a variety of computer vision datasets, but it currently struggles with respect to certain tasks such as fine-grained classification and counting objects. The model also poses issues with regards to fairness and bias, and the specific biases it exhibits can depend significantly on class design and the choices one makes for categories to include and exclude.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-zero-shot-image-classification" target="_blank">zero-shot-image-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-zero-shot-image-classification" target="_blank">zero-shot-image-classification-online-endpoint.sh</a>
Batch|<a href="https://aka.ms/azureml-infer-batch-sdk-zero-shot-image-classification" target="_blank">zero-shot-image-classification-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-zero-shot-image-classification" target="_blank">zero-shot-image-classification-batch-endpoint.sh</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
   "input_data":{
      "columns":[
         "image", "text"
      ],
      "index":[0, 1],
      "data":[
         ["image1", "label1, label2, label3"],
         ["image2"]
      ]
   }
}
```
Note:
- "image1" and "image2" should be publicly accessible urls or strings in `base64` format.
- The text column in the first row determines the labels for image classification. The text column in the other rows is not used and can be blank.

#### Sample output

```json
[
    {
        "probs": [0.95, 0.03, 0.02],
        "labels": ["label1", "label2", "label3"]
    },
    {
        "probs": [0.04, 0.93, 0.03],
        "labels": ["label1", "label2", "label3"]
    }
]
```

#### Model inference - visualization
For a sample image and label text "credit card payment, contactless payment, cash payment, mobile order".

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_openai-clip-vit-base-patch32_cafe_ZSIC.jpg" alt="zero shot image classification visualization">

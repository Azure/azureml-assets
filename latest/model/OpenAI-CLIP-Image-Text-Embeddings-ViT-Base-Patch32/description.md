The `CLIP` model was developed by OpenAI researchers to learn about what contributes to robustness in computer vision tasks and to test the ability of models to generalize to arbitrary image classification tasks in a zero-shot manner. The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. The model was trained on publicly available image-caption data, which was gathered in a mostly non-interventionist manner. The model is intended as a research output for research communities, and the primary intended users of these models are AI researchers. The model has been evaluated on a wide range of benchmarks across a variety of computer vision datasets, but it currently struggles with respect to certain tasks such as fine-grained classification and counting objects. The model also poses issues with regards to fairness and bias, and the specific biases it exhibits can depend significantly on class design and the choices one makes for categories to include and exclude.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-text-embeddings" target="_blank">image-text-embeddings-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-text-embeddings" target="_blank">image-text-embeddings-online-endpoint.sh</a>
Batch|<a href="https://aka.ms/azureml-infer-batch-sdk-image-text-embeddings" target="_blank">image-text-embeddings-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-image-text-embeddings" target="_blank">image-text-embeddings-batch-endpoint.sh</a>

### Sample inputs and outputs (for real-time inference)

#### Sample input for image embeddings

```json
{
   "input_data":{
      "columns":[
         "image", "text"
      ],
      "index":[0, 1],
      "data":[
         ["image1", ""],
         ["image2", ""]
      ]
   }
}
```
Note: "image1" and "image2" should be publicly accessible urls or strings in `base64` format

#### Sample output

```json
[
    {
        "image_features": [-0.92, -0.13, 0.02, ... , 0.13],
    },
    {
        "image_features": [0.54, -0.83, 0.13, ... , 0.26],
    }
]
```
Note: returned embeddings have dimension 512 and are not normalized

#### Sample input for text embeddings

```json
{
   "input_data":{
      "columns":[
         "image", "text"
      ],
      "index":[0, 1],
      "data":[
         ["", "sample text 1"],
         ["", "sample text 2"]
      ]
   }
}
```

#### Sample output

```json
[
    {
        "text_features": [0.42, -0.13, -0.92, ... , 0.63],
    },
    {
        "text_features": [-0.14, 0.93, -0.15, ... , 0.66],
    }
]
```
Note: returned embeddings have dimension 512 and are not normalized

#### Sample input for image and text embeddings

```json
{
   "input_data":{
      "columns":[
         "image", "text"
      ],
      "index":[0, 1],
      "data":[
         ["image1", "sample text 1"],
         ["image2", "sample text 2"]
      ]
   }
}
```
Note: "image1" and "image2" should be publicly accessible urls or strings in `base64` format

#### Sample output

```json
[
    {
        "image_features": [0.92, -0.13, 0.02, ... , -0.13],
        "text_features": [0.42, 0.13, -0.92, ... , -0.63]
    },
    {
        "image_features": [-0.54, -0.83, 0.13, ... , -0.26],
        "text_features": [-0.14, -0.93, 0.15, ... , 0.66]
    }
]
```
Note: returned embeddings have dimension 512 and are not normalized
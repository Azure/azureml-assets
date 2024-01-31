The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a self-supervised fashion.

Images are presented to the model as a sequence of fixed-size patches, which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

Note that this model does not include any fine-tuned heads.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

### Intended uses & limitations
You can use the raw model for feature extraction.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-embeddings" target="_blank">image-embeddings-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-embeddings" target="_blank">image-embeddings-online-endpoint.sh</a>

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
Note: "image1" and "image2" should be publicly accessible urls or strings in `base64` format.

#### Sample output

```json
[
    {
        "image_features": [0.55, 0.32, -0.82, ... , 0.29],
    },
    {
        "image_features": [-0.36, -0.97, 0.43, ... , 0.11],
    }
]
```
Note: returned features have dimension 768 and are not normalized.

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a self-supervised fashion with the DinoV2 method.

Images are presented to the model as a sequence of fixed-size patches, which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

Note that this model does not include any fine-tuned heads.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

### Limitations and Biases

Despite improvements thanks to the training method not using annotations, we still observe significant biases in our models toward rich households from Western countries. We expect fine-tuning will increase the biases in the features produced by the model as they will be tuned to the fine-tuning labels.

### License

Apache License 2.0

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-image-embeddings" target="_blank">image-embeddings-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-image-embeddings" target="_blank">image-embeddings-online-endpoint.sh</a>

### Sample input and output

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
        "image_features": [0.91, -0.64, 0.17, ... , -0.35],
    },
    {
        "image_features": [0.78, 0.04, 0.22, ... , -0.61],
    }
]
```
Note: returned features have dimension 1536 and are not normalized.

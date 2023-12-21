The `CLIP` model was developed by researchers at OpenAI to learn about what contributes to robustness in computer vision tasks. The model was also developed to test the ability of models to generalize to arbitrary image classification tasks in a zero-shot manner. It was not developed for general model deployment - to deploy models like `CLIP, researchers will first need to carefully study their capabilities in relation to the specific context they’re being deployed within.

This model uses a ViT-L/14 Transformer architecture trained at 336x336 pixel resolution as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.

The primary intended users of these models are AI researchers for tasks requiring image and/or text embeddings such as text and image retrieval.

# Training Details

## Training Data
The model was trained on publicly available image-caption data. This was done through a combination of crawling a handful of websites and using commonly-used pre-existing image datasets such as YFCC100M. A large portion of the training data comes from the authors' crawling of the internet. This means that the data is more representative of people and societies most connected to the internet which tend to skew towards more developed nations, and younger, male users.

# Evaluation Results
This model was evaluated for text retrieval and image retrieval tasks on the Flickr30k and MSCOCO datasets. The results from Table 13 in the [original CLIP paper](https://arxiv.org/pdf/2103.00020.pdf) are summarized below

Text Retrieval
|Dataset| R@1 | R@5 |
|-------|-----|-----|
| Flickr30k| 88.0 | 98.7 |
| MSCOCO | 58.4 | 81.5 |

Image Retrieval
|Dataset| R@1 | R@5 |
|-------|-----|-----|
| Flickr30k| 68.7 | 90.6 |
| MSCOCO | 37.8 | 62.4 |

# Limitations and Biases

## Bias
The authors of the [original CLIP paper](https://arxiv.org/pdf/2103.00020.pdf) found that the performance of the model and its biases can depend significantly on class design and the choices one makes for categories to include and exclude. They tested the risk of certain kinds of denigration with CLIP by classifying images of people from Fairface into crime-related and non-human animal categories. They found significant disparities with respect to race and gender, which could shift based on how the classes were constructed. The authors also tested the performance of CLIP on gender, race, and age classification using the Fairface dataset. They found that the accuracy for gender classification was greater than 96% across all races, with 'Middle Eastern' having the highest accuracy (98.4%) and 'White' having the lowest (96.5%). Additionally, CLIP averaged ~93% for racial classification and ~63% for age classification.

# License
The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). The license allows users to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the conditions that the copyright notice and permission notice appear in all copies or substantial portions of the software.


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
Note: returned embeddings have dimension 768 and are not normalized

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
Note: returned embeddings have dimension 768 and are not normalized

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
Note: returned embeddings have dimension 768 and are not normalized
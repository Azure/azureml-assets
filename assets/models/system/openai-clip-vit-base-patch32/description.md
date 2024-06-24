OpenAI's CLIP (Contrastive Language–Image Pre-training) model was designed to investigate the factors that contribute to the robustness of computer vision tasks. It can seamlessly adapt to a range of image classification tasks without requiring specific training for each, demonstrating efficiency, flexibility, and generality.

In terms of architecture, CLIP utilizes a ViT-B/32 Transformer for image encoding and a masked self-attention Transformer for text encoding. These encoders undergo training to improve the similarity of (image, text) pairs using a contrastive loss.

For training purposes, CLIP leverages image-text pairs from the internet and engages in a proxy task: when presented with an image, predict the correct text snippet from a set of 32,768 randomly sampled options. This approach allows CLIP to comprehend visual concepts and establish associations with their textual counterparts, enhancing its performance across various visual classification tasks.

The design of CLIP effectively tackles notable challenges, including the dependence on expensive labeled datasets, the need for fine-tuning on new datasets to achieve optimal performance across diverse tasks, and the disparity between benchmark and real-world performance.

The primary intended users of these models are AI researchers for tasks requiring image and/or text embeddings such as text and image retrieval.

For more details on CLIP model, review the <a href="https://arxiv.org/abs/2103.00020" target="_blank">original-paper</a> or the <a href="https://github.com/openai/CLIP/blob/main/model-card.md" target="_blank">original-model-card</a>.

# Training Details

## Training Data

The training of the CLIP model involved utilizing publicly accessible image-caption data obtained by crawling several websites and incorporating commonly-used existing image datasets like YFCC100M. Researchers curated a novel dataset comprising 400 million image-text pairs sourced from diverse publicly available internet outlets. This dataset, referred to as WIT (WebImageText), possesses a word count comparable to the WebText dataset employed in training GPT-2.

As a consequence, the data in WIT is reflective of individuals and societies predominantly linked to the internet, often leaning towards more developed nations and a demographic skewed towards younger, male users.

## Training Procedure

The Vision Transformers ViT-B/32 underwent training for 32 epochs, employing the Adam optimizer with applied decoupled weight decay regularization. The learning rate was decayed using a cosine schedule. The learnable temperature parameter τ was initialized to the equivalent of 0.07. Training utilized a very large mini-batch size of 32,768, and mixed-precision techniques were employed to expedite training and conserve memory. The largest Vision Transformer was trained over a period of 12 days on 256 V100 GPUs. For a more in-depth understanding, refer to sections 2 and 3 of the <a href="https://arxiv.org/abs/2103.00020" target="_blank">original-paper</a>.

# Evaluation Results

The performance of CLIP has been evaluated on a wide range of benchmarks across a variety of computer vision datasets such as OCR to texture recognition to fine-grained classification. The section 3 and 4 of the paper describes model performance on multiple datasets.

# Limitations and Biases

CLIP has difficulties with tasks such as fine-grained classification and object counting. Its performance also raises concerns regarding fairness and bias. Additionally, there is a notable limitation in the evaluation approach, with the use of linear probes potentially underestimating CLIP's true performance, as suggested by evidence.

CLIP's performance and inherent biases can vary depending on class design and category choices. Assessing Fairface images unveiled significant racial and gender disparities, influenced by class construction. Evaluations on gender, race, and age classification using the Fairface dataset indicated gender accuracy exceeding 96%, with variations among races. Racial classification achieved approximately 93%, while age classification reached around 63%. These assessments aim to gauge model performance across demographics, pinpoint potential risks, and are not intended to endorse or promote such tasks. For a more details, refer to sections 6 and 7 of the <a href="https://arxiv.org/abs/2103.00020" target="_blank">original-paper</a>.

# License

MIT License

### Inference Samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-sdk-zero-shot-image-classification" target="_blank">zero-shot-image-classification-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-cli-zero-shot-image-classification" target="_blank">zero-shot-image-classification-online-endpoint.sh</a>
Batch|<a href="https://aka.ms/azureml-infer-batch-sdk-zero-shot-image-classification" target="_blank">zero-shot-image-classification-batch-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-batch-cli-zero-shot-image-classification" target="_blank">zero-shot-image-classification-batch-endpoint.sh</a>

# Sample input and output

### Sample input

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

### Sample output

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

#### Visualization of inference result for a sample image

For a sample image and label text "credit card payment, contactless payment, cash payment, mobile order".

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/plot_openai-clip-vit-base-patch32_cafe_ZSIC.jpg" alt="zero shot image classification visualization">

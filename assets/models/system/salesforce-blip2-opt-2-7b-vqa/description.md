The `BLIP-2` model, utilizing OPT-2.7b (a large language model with 2.7 billion parameters), is presented in the <a href='https://arxiv.org/abs/2201.12086' target='_blank'>paper</a> titled "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models". This is a generic and efficient pre-training strategy that easily harvests development of pre-trained vision models and large language models (LLMs) for Vision-Language Pre-training (VLP). This model was made available in this <a href='https://github.com/salesforce/LAVIS/tree/main/projects/blip2' target='_blank'>repository</a>.

BLIP-2 consists of 3 models: a CLIP-like image encoder, a Querying Transformer (Q-Former) and a large language model. The authors initialize the weights of the image encoder and large language model from pre-trained checkpoints and keep them frozen while training the Querying Transformer, which is a BERT-like Transformer encoder that maps a set of "query tokens" to query embeddings, which bridge the gap between the embedding space of the image encoder and the large language model.

The model's objective is to predict the next text token based on query embeddings and the previous text. This functionality allows the model to undertake a range of tasks, such as generating image captions, responding to visual questions (VQA), and participating in chat-like conversations using the image and preceding chat as input prompts.

# Limitations and Biases

BLIP2-OPT uses off-the-shelf OPT as the language model. It shares the same potential risks and limitations outlined in Meta's model card.

> Like other large language models for which the diversity (or lack thereof) of training data induces downstream impact on the quality of our model, OPT-175B has limitations in terms of bias and safety. OPT-175B can also have quality issues in terms of generation diversity and hallucination. In general, OPT-175B is not immune from the plethora of issues that plague modern large language models.

BLIP2 undergoes fine-tuning on internet collected image-text datasets, which raises concerns about potential inappropriate content generation or replicating inherent biases from the underlying data. The model has not been tested in real-world applications, and caution is advised against direct deployment. Researchers should carefully assess the model's safety and fairness in the specific deployment context before considering its use.

# License

mit

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
         ["image1", "What is in the picture? Answer: "],
         ["image2", "what are people doing? Answer: "]
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
      "text": "a stream in the desert"
   },
   {
      "text": "they're buying coffee"
   }
]
```

#### Visualization of inference result for a sample image

For sample image below and text prompt "what are people doing? Answer: ", the output text is "they're buying coffee".

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_blip2_vqa.png" alt="Salesforce-BLIP2-vqa">
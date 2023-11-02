`BLIP-2` is a model consisting of three components: a CLIP-like image encoder, a Querying Transformer (Q-Former), and a large language model. The image encoder and language model are initialized from pre-trained checkpoints and kept frozen while training the Querying Transformer. The model's goal is to predict the next text token given query embeddings and previous text, making it useful for tasks such as image captioning, visual question answering, and chat-like conversations. However, the model inherits the same risks and limitations as the off-the-shelf OPT language model it uses, including bias, safety issues, generation diversity issues, and potential vulnerability to inappropriate content or inherent biases in the underlying data. Researchers should carefully assess the safety and fairness of the model before deploying it in any real-world applications.

> The above summary was generated using ChatGPT. Review the <a href='https://huggingface.co/Salesforce/blip2-opt-2.7b' target='_blank'>original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
         ["image1", "What is in the picture? Answer: "],
         ["image2", "what are people doing? Answer: "]
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
      "text": "a stream in the desert"
   },
   {
      "text": "they're buying coffee"
   }
]
```

#### Model inference - visual question answering
For a sample image and text prompt "what are people doing? Answer: ".

<img src="https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/output_blip2_vqa.png" alt="Salesforce-BLIP2-vqa">
Multimodal Early Fusion Transformer, MMEFT, is a transformer-based model tailored for processing both structured and unstructured data.

It can be used for multi-class and multi-label multimodal classification tasks, and is capable of handling datasets with features from diverse modes, including categorical, numerical, image, and text.
The MMEFT architecture is composed of embedding, fusion, aggregation, and output layers. The embedding layer produces independent non-contextual embeddings for features of varying modes. Then, the fusion Layer integrates the non-contextual embeddings to yield contextual multimodal embeddings. The aggregation layer consolidates these contextual multimodal embeddings into a single multimodal embedding vector. Lastly, the output Layer, processes the final multimodal embedding to generate the model's prediction based on task for which it is used. 
MMEFT uses BertTokenizer for text data embeddings, and considers 'openai/clip-vit-base-patch32' model from Hugging Face for image data embeddings.
This model is designed to offer a comprehensive approach to multimodal data, ensuring accurate and efficient classification across varied datasets.
NOTE: We highly recommend to finetune the model on your dataset before deploying.
 
 ### Inference samples 
  
 Inference type|Python sample (Notebook)|CLI with YAML 
 |--|--|--| 
 Real time|<a href='https://aka.ms/azureml-infer-sdk-multimodal-classification' target='_blank'>multimodal-classification-online-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-cli-multimodal-classification' target='_blank'>multimodal-classification-online-endpoint.sh</a> 
 Batch |<a href='https://aka.ms/azureml-infer-batch-sdk-multimodal-classification' target='_blank'>multimodal-classification-batch-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-batch-cli-multimodal-classification' target='_blank'>multimodal-classification-batch-endpoint.sh</a> 
 
 ### Finetuning samples 
  
 Task|Dataset|Python sample (Notebook)|CLI with YAML 
 |--|--|--|--| 
 Multimodal multi-class classification|[Airbnb listings dataset](https://automlresources-prod.azureedge.net/datasets/AirBnb.zip)|<a href='https://aka.ms/azureml-ft-sdk-multimodal-mc-classification' target='_blank'>multimodal-multiclass-classification.ipynb</a>|<a href='https://aka.ms/azureml-ft-cli-multimodal-mc-classification' target='_blank'>multimodal-multiclass-classification.sh</a> 
 Multimodal multi-label classification |[Chest X-Rays dataset](https://automlresources-prod.azureedge.net/datasets/ChXray.zip)|<a href='https://aka.ms/azureml-ft-sdk-multimodal-ml-classification' target='_blank'>multimodal-multilabel-classification.ipynb</a>|<a href='https://aka.ms/azureml-ft-cli-multimodal-ml-classification' target='_blank'>multimodal-multilabel-classification.sh</a> 
 
### Sample inputs and outputs (for real-time inference) 

#### Sample input 

```json 
{ 
 "input_data": { 
        "columns": ["column1","column2","column3","column4","column5","column6"], 
        "data": [[22,11.2,"It was a great experience!",image1,"Categorical value",True],
                 [111,8.2,"I may not consider this option again.",image2,"Categorical value",False]
                ]
     } 
} 
``` 

> Note:
>
> - "image1", "image2" are strings in base64 format.
  
#### Sample output 
  
  
```json 
[ 
     {
        "label1": 0.1,
        "label2": 0.7,
        "label3": 0.2
     }, 
     {
        "label1": 0.3,
        "label2": 0.3,
        "label3": 0.4
     },
] 
  
```

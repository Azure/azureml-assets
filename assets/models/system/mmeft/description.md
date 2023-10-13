Multimodal Early Fusion Transformer (MMEFT): A Transformer based Architecture for Reasoning over Structured and Unstructured Data

MMEFT model can be used for multi-class and multi-label multimodal classification tasks. Dataset can have features belonging to one of the modes in {categorical,numerical,image,text}.
The MMEFT architecture is composed of embedding, fusion, aggregation, and output layers. The embedding layer creates independent non-contextual embeddings for features of varying modes. The fusion layer inputs these non-contextual embeddings and outputs a set of contextual multimodal embeddings. These are aggregated into a single multimodal embedding in the aggregation layer. The multimodal embedding is then passed through a task-specific output layer which outputs the model’s estimate. 
BertTokenizer is used to get embeddings for text data. 'openai/clip-vit-base-patch32' model from Hugging Face is used for image embeddings.
  
 ### Inference samples 
  
 Inference type|Python sample (Notebook)|CLI with YAML 
 |--|--|--| 
 Real time|<a href='https://aka.ms/azureml-infer-sdk-multimodal-classification' target='_blank'>multimodal-classification-online-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-cli-multimodal-classification' target='_blank'>multimodal-classification-online-endpoint.sh</a> 
 Batch |<a href='https://aka.ms/azureml-infer-batch-sdk-multimodal-classification' target='_blank'>multimodal-classification-batch-endpoint.ipynb</a>|<a href='https://aka.ms/azureml-infer-batch-cli-multimodal-classification' target='_blank'>multimodal-classification-batch-endpoint.sh</a> 
 
 ### Finetuning samples 
  
 Task|Dataset|Python sample (Notebook)|CLI with YAML 
 |--|--|--|--| 
 Multimodal multi-class classification|TODO: add dataset|<a href='https://aka.ms/azureml-ft-sdk-multimodal-mc-classification' target='_blank'>multimodal-multiclass-classification.ipynb</a>|<a href='https://aka.ms/azureml-ft-cli-multimodal-mc-classification' target='_blank'>multimodal-multiclass-classification.sh</a> 
 Multimodal multi-label classification |TODO: add dataset|<a href='https://aka.ms/azureml-ft-sdk-multimodal-ml-classification' target='_blank'>multimodal-multilabel-classification.ipynb</a>|<a href='https://aka.ms/azureml-ft-cli-multimodal-ml-classification' target='_blank'>multimodal-multilabel-classification.sh</a> 
 
 ### Sample inputs and outputs (for real-time inference) 
 #### Sample input 
 ```json 
 { 
 'input_data': { 
         'columns': ['column1','column2','column3','column4','column5','column6'], 
         'data': [22,11.2,'It was a great experience!',<base 64 encoded image string>,'Categorical value',True]
     } 
 } 
 ``` 
  
 #### Sample output 
  
  
 ```json 
 [ 
     { 
         'label1': 0.1,
		 'label2': 0.7,
		 'label3': 0.2
     } 
 ] 
  
 ``` 
  
 #### Model inference - visualization for a sample image 
  
 <img src='https://automlcesdkdataresources.blob.core.windows.net/finetuning-image-models/images/Model_Result_Visualizations(Do_not_delete)/TODO' alt='mc visualization'> 

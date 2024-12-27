# **Model Details**

Name: bge-small-en-v1.5
FlagEmbedding is a versatile tool designed to map text into low-dimensional dense vectors, enabling applications in retrieval, classification, clustering, and semantic search. It proves valuable for tasks related to Language Model Models (LLMs) and can be utilized in vector databases.

Updates:

10/12/2023: LLM-Embedder release, offering unified embedding for diverse retrieval needs in LLMs.

09/15/2023: Release of the technical report and massive training data for BGE (FlagEmbedding).

09/12/2023: Introduction of new models - cross-encoder models BAAI/bge-reranker-base and BAAI/bge-reranker-large, recommended for re-ranking top-k documents.

BGE Versions (v1.5): Addressing similarity distribution issues, enhancing retrieval ability without instruction, and offering improved performance.

BAAI Embedding:
Pre-training models using retromae, followed by large-scale pairs data training using contrastive learning. Fine-tuning the embedding model on specific data is possible, and pre-trained models are provided. Pre-training focuses on text reconstruction.

BGE Reranker:
Uses cross-encoder for full-attention over input pairs, balancing accuracy and time cost. Suited for re-ranking top-k documents returned by embedding models. Trained on multilingual pair data, with data format similar to the embedding model.

License:
FlagEmbedding is licensed under the MIT License, allowing for free commercial use of the released models.

In essence, FlagEmbedding provides a powerful and adaptable solution for text mapping, with recent updates, additional models, and licensing flexibility.

For More Details: https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md


# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# **Sample inputs and outputs**

### **Sample input**
```json
{
    "input_data":{
       "input_string":["the meaning of life is"],
       "parameters":{
             "temperature":0.5,
             "top_p":0.5,
             "max_new_tokens":100,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is satellite impossible se lee designsser stickwehr intend doing android random ghz willingness bones reviews darfer pixel random ghz ghz ghz spacecraft android hardware androidwskiord fingertips regretted se surprise glove beatles android randomsser lennon cookies android schmidtul android sm bandsk androidsser glove intentional schmidt parker mustard rao teammate counties random orbiting parker mustard exact drummer danny sovietsk pointless bone lp sm firearm yourselfssersk sm sm cheap fixsser photography steve sm fitlogram actually counties sm sm band random bone engineers android smlogram intentionalsser yourselfssersk"
  }
]
```

# **Model Details**

The "BAAI/bge-small-en" model is based on the given information, here's a summary of the relevant details:

FlagEmbedding Overview:

Focuses on retrieval-augmented Language Models (LLMs).
Projects include Fine-tuning of LM (LM-Cocktail), Dense Retrieval (LLM Embedder, BGE Embedding, C-MTEB), and Reranker Model (BGE Reranker).
News and Releases:

11/23/2023: Released LM-Cocktail, a method for fine-tuning language models by merging multiple models. Technical Report provided.
10/12/2023: Released LLM-Embedder, a unified embedding model supporting diverse retrieval augmentation needs. Technical Report available.
09/15/2023: Released the technical report and massive training data of BGE (not specifying the "BAAI/bge-small-en" model).
09/12/2023: Introduced new reranker models (BAAI/bge-reranker-base and BAAI/bge-reranker-large) more powerful than the embedding model. Recommended for re-ranking top-k documents returned by embedding models. Updated the embedding model (bge-*-v1.5) to address similarity distribution issues and enhance retrieval ability without explicit instruction.
More Information:

Model List:
"bge" stands for BAAI general embedding.
No specific details about the "BAAI/bge-small-en" model are provided in the summary.
To get specific details about the "BAAI/bge-small-en" model, it's recommended to refer to the FlagEmbedding GitHub repository or additional documentation provided by the project. The summary focuses on the information available in the provided text.

More Information : https://github.com/FlagOpen/FlagEmbedding

# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


## **Sample inputs and outputs (for real-time inference)**

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
    "0": "the meaning of life is strip lgbt 40 40 layered strip resume 40 layers 40 40 40 layers 40ipip anymore hip happinessip 25 40 layers 40 40 labourip more relaxation housing relax politicians relaxationage layers fade 40 40 quota floor relaxation temporary relaxation temporary pad happiness 40 programme labour relax temporary bed relaxation party 40 wing happiness 40 escape leisure relaxation temporary 40ip 40 parliament saturday relaxation dams 40 labour relaxationage relaxation dams hr 40 relaxation dams pet relaxation temporary boredvill strip smile strip dominate compromise labour 40 layers leisure dams 40 25 layers strip dominate strip"
  }
]
```

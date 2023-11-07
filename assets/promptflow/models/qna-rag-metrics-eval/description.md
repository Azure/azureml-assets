The Q&A RAG (Retrieval Augmented Generation) evaluation flow will evaluate the Q&A RAG systems by leveraging the state-of-the-art Large Language Models (LLM) to measure the quality and safety of your responses . Utilizing GPT model to assist with measurements aims to achieve a high agreement with human evaluations compared to traditional mathematical measurements.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://github.com/microsoft/promptflow/blob/pm/3p-inside-materials/docs/media/deploy-to-aml-code/sdk/deploy.ipynb" target="_blank">deploy-promptflow-model-python-example</a>|<a href="https://github.com/microsoft/promptflow/blob/pm/3p-inside-materials/docs/go-to-production/deploy-to-aml-code.md" target="_blank">deploy-promptflow-model-cli-example</a>
Batch | N/A | N/A

### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "question": "What is the purpose of the LLM Grounding Score, and what does a higher score mean in this context?",
        "answer": "The LLM Grounding Score gauges an LLM's grasp of provided context in in-context learning. A higher score implies better understanding and more accurate responses.",
        "metrics": "gpt_groundedness,gpt_retrieval_score,gpt_relevance",
        "documents": "{'documents': [{'[doc1]': {'title': 'In-Context Learning with Large-Scale Pretrained Language Models',\r'content': 'In-Context Learning uses large pretrained models to acquire new skills. GPT-3 introduced this, achieving accuracy similar to fine-tuned models. Prompt order and similar training examples affect performance. Retrievers locate exemplary few-shot examples, with semantic similarity fine-tuning. Advanced retriever use includes code generation, but 'fantastic' examples assumption has task-specific limitations.'}}]}"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "gpt_groundedness":5,
        "gpt_relevance":5,
        "gpt_retrieval_score":1
    }
}
```
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
        "answer": "The LLM Grounding Score is a metric used in the context of in-context learning with large-scale pretrained language models (LLMs) [doc1]. It measures the ability of the LLM to understand and connect with the provided context or demonstrations during the learning process.A higher LLM Grounding Score indicates that the LLM has a better understanding and connection with the context or demonstrations [doc1]. It means that the LLM is able to effectively utilize the provided examples and generate more accurate and relevant responses based on that context.Please note that the LLM Grounding Score is specific to the in-context learning framework and may not be applicable in other contexts or tasks.[doc1]: In-Context Learning with Large-Scale Pretrained Language Models: How Far Are We? (2022) - zelin, English.",
        "metrics": "gpt_groundedness,gpt_retrieval_score,gpt_relevance",
        "documents": "{'documents': [{'[doc1]': {'title': 'In-Context Learning with Large-Scale Pretrained Language Models',\r'content': 'In-Context Learning, different from Few-Shot Learning, leverages large pretrained language models to acquire new skills with minimal training examples. GPT-3 introduced this concept, achieving accuracy comparable to fine-tuned models. The order of prompts and selecting similar training examples significantly impact performance. To enhance in-context learning, \"Retrievers\" are used to find exemplary few-shot examples. Retrievers, fine-tuned for semantic similarity, excel in this role. For advanced applications, retrievers can be fine-tuned for downstream tasks, such as code generation. However, the assumption that training examples with the same structure are always \"fantastic\" has limitations, necessitating different approaches for various tasks.'}}]}"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "gpt_groundedness":5,
        "gpt_relevance":"NaN",
        "gpt_retrieval_score":5
    }
}
```
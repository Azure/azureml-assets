The Q&A evaluation flow will evaluate the Q&A systems by leveraging the state-of-the-art Large Language Models (LLM) to measure the quality and safety of your responses. Utilizing GPT and GPT embedding model to assist with measurements aims to achieve a high agreement with human evaluations compared to traditional mathematical measurements.


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
        "question": "Which camping table holds the most weight?",
        "answer": "The Alpine Explorer Tent is the most waterproof.",
        "context": "From the our product list, the alpine explorer tent is the most waterproof. The Adventure Dining Tabbe has higher weight.",
        "ground_truth": "The Alpine Explorer Tent has the highest rainfly waterproof rating at 3000m",
        "metrics": "gpt_groundedness,f1_score,ada_similarity,gpt_fluency,gpt_coherence,gpt_similarity,gpt_relevance"
    }
}
```

#### Sample output
```json
{
    "outputs": {      
        "f1_score":0.5,
        "gpt_coherence":1,
        "gpt_similarity":1,
        "gpt_fluency":1,
        "gpt_relevance":1,
        "gpt_groundedness":5,
        "ada_similarity":0.9317354400079281
    }
}
```
The Q&A Similarity evaluation flow will evaluate the Q&A Retrieval Augmented Generation systems by leveraging the state-of-the-art Large Language Models (LLM) to measure the quality and safety of your responses. Utilizing GPT as the Language Model to assist with measurements aims to achieve a high agreement with human evaluations compared to traditional mathematical measurements.


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
        "question": "What feeds all the fixtures in low voltage tracks instead of each light having a line-to-low voltage transformer?",
        "ground_truth": "Master transformer.",
        "answer": "The main transformer is the object that feeds all the fixtures in low voltage tracks."
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "gpt_similarity": 4
    }
}
```
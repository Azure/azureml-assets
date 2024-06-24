The "QnA Ada Similarity Evaluation" is a model to evaluate the Q&A Retrieval Augmented Generation systems by leveraging the state-of-the-art Large Language Models (LLM) to measure the quality and safety of your responses. Utilizing GPT-3.5 as the Language Model to assist with measurements aims to achieve a high agreement with human evaluations compared to traditional mathematical measurements.


### Inference samples

Inference type|CLI|VS Code Extension
|--|--|--|
Real time|<a href="https://microsoft.github.io/promptflow/how-to-guides/deploy-a-flow/index.html" target="_blank">deploy-promptflow-model-cli-example</a>|<a href="https://microsoft.github.io/promptflow/how-to-guides/deploy-a-flow/index.html" target="_blank">deploy-promptflow-model-vscode-extension-example</a>
Batch | N/A | N/A

### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "ground_truth": "Master transformer.",
        "answer": "The main transformer is the object that feeds all the fixtures in low voltage tracks."
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "ada_similarity": 0.870509349428722
    }
}
```
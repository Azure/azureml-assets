The "Bring Your Own Data Chat QnA" is a pre-trained chat model, enhanced by GPT3.5, that leverages your personally indexed data and chat history to deliver more concrete and relevant answers. It involves processing the raw query through an embedding procedure, followed by a "Vector Search" to pinpoint the most pertinent context within the user's data. Subsequently, GPT3.5 is employed to generate a comprehensive answer to the question using the sourced documents.


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
        "question": "What is Azure compute instance?"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "answer": "An Azure Machine Learning compute instance is a fully managed cloud-based workstation for data scientists. It provides a pre-configured and managed development environment in the cloud for machine learning. Compute instances can also be used as a compute target for training and inferencing for development and testing purposes. They have a job queue, run jobs securely in a virtual network environment, and can run multiple small jobs in parallel. Additionally, compute instances support single-node multi-GPU distributed training jobs."
    }
}
```
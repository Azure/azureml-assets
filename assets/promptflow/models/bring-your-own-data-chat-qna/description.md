This sample demonstrates multi-round Q&A chatbot powered by GPT. It utilizes indexed files from Azure Machine Learning to provide grounded answers. You can ask a wide range of questions related to Azure Machine Learning and receive responses. The process involves embedding the raw query, using vector search to find most relevant context in user data, and then using GPT to chat with you with the documents. This sample also contains multiple prompt variants that you can tune.


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
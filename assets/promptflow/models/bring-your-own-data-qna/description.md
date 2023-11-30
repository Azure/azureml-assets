This sample demonstrates Q&A application powered by GPT. It utilizes indexed files from Azure Machine Learning to provide grounded answers. You can ask a wide range of questions related to Azure Machine Learning and receive responses. The process involves embedding the raw query, using vector search to find most relevant context in user data, and then using GPT to chat with you with the documents. This sample also contains multiple prompt variants that you can tune.


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
        "question": "How to use SDK V2?"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "output": "To use the Azure Machine Learning Python SDK v2, you need to have an Azure Machine Learning workspace and the SDK installed. You can either create a compute instance, which automatically installs the SDK and is pre-configured for ML workflows, or use the provided commands to install the SDK. (Source: https://github.com/prakharg-msft/azureml-tutorials/blob/main//how-to-auto-train-image-models.md)"
    }
}
```
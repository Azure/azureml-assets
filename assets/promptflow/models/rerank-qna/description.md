WIP


### Inference samples

Inference type|CLI|VS Code Extension
|--|--|--|
Real time|<a href="https://microsoft.github.io/promptflow/how-to-guides/deploy-a-flow/index.html" target="_blank">deploy-promptflow-model-cli-example</a>|<a href="https://microsoft.github.io/promptflow/how-to-guides/deploy-a-flow/index.html" target="_blank">deploy-promptflow-model-vscode-extension-example</a>
Batch | N/A | N/A

### Sample inputs and outputs (for real-time inference) (WIP)

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
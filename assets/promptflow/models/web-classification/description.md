The "Web Classification" is a model demonstrating multi-class classification with LLM. Given an url, it will classify the url into one web category with just a few shots, simple summarization and classification prompts.


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
        "url": "https://www.microsoft.com/en-us/store/collections/xboxseriessconsoles?icid=CNav_Xbox_Series_S"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "category": "App",
        "evidence": "Both"
    }
}
```
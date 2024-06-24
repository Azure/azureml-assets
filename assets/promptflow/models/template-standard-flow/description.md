The "Template Standard Flow" is a model using GPT3.5 to generate a joke based on user input.


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
        "topic": "atom"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "joke": "Sure, here you go: Why can't you trust an atom? Because they make up everything!"
    }
}
```
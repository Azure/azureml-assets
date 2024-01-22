The "Template Evaluation Flow" is a evaluate model to measure how well the output matches the expected criteria and goals.


### Inference samples

Inference type|CLI|VS Code Extension
|--|--|--|
Real time|<a href="https://microsoft.github.io/promptflow/how-to-guides/deploy-a-flow/index.html" target="_blank">deploy-promptflow-model-cli-example</a>|<a href="https://microsoft.github.io/promptflow/how-to-guides/deploy-a-flow/index.html" target="_blank">deploy-promptflow-model-python-example</a>
Batch | N/A | N/A

### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "groundtruth": "Tomorrow's weather will be sunny.",
        "prediction": "The weather will be sunny tomorrow."
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "results": {}
    }
}
```
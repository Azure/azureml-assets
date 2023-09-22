The "Template Evaluation Flow" is a evaluate model to measure how well the output matches the expected criteria and goals.


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
The "Classification Accuracy Evaluation" is a model designed to assess the effectiveness of a data classification system. It involves matching each prediction against the ground truth, subsequently assigning a "Correct" or "Incorrect" score. The cumulative results are then leveraged to generate performance metrics, such as accuracy, providing an overall measure of the system's proficiency in data classification.


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
        "groundtruth": "App",
        "prediction": "App"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "grade": "Correct"
    }
}
```
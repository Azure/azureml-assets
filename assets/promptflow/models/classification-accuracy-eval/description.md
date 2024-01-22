The "Classification Accuracy Evaluation" is a model designed to assess the effectiveness of a data classification system. It involves matching each prediction against the ground truth, subsequently assigning a "Correct" or "Incorrect" score. The cumulative results are then leveraged to generate performance metrics, such as accuracy, providing an overall measure of the system's proficiency in data classification.


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
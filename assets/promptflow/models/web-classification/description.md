This sample demonstrates multi-class classification with LLM. Given an URL, it will classify the URL into one web category with just a few shots, simple summarization and classification prompts.


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
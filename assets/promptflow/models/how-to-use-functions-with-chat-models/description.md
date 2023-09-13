The "How to Use Functions with Chat Models" is a chat model illustrates how to employ the LLM tool's Chat API with external functions, thereby expanding the capabilities of GPT models. The Chat Completion API includes an optional 'functions' parameter, which can be used to stipulate function specifications. This allows models to generate arguments that comply with the given specifications. However, it's important to note that the API will not directly execute any function calls. The responsibility of executing function calls using the model outputs lies with the developers.


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
        "question": "How about London next week?"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "answer": "Function generation requested, function = get_n_day_weather_forecast, args = { 'location': 'London', 'num_days': 7, 'format': 'celsius' }"
    }
}
```
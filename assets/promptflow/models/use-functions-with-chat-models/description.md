This flow covers how to use the LLM tool chat API in combination with external functions to extend the 
capabilities of GPT models. 

`functions` is an optional parameter in the <a href='https://platform.openai.com/docs/api-reference/chat/create' target='_blank'>Chat Completion API</a> which can be used to provide function 
specifications. The purpose of this is to enable models to generate function arguments which adhere to the provided 
specifications. Note that the API will not actually execute any function calls. It is up to developers to execute 
function calls using model outputs. 

If the `functions` parameter is provided then by default the model will decide when it is appropriate to use one of the 
functions. The API can be forced to use a specific function by setting the `function_call` parameter to 
`{"name": "<insert-function-name>"}`. The API can also be forced to not use any function by setting the `function_call` 
parameter to `"none"`. If a function is used, the output will contain `"finish_reason": "function_call"` in the 
response, as well as a `function_call` object that has the name of the function and the generated function arguments. 
You can refer to <a href='https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb' target='_blank'>openai sample</a> for more details.


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
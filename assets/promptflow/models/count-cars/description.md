The "Count Cars" is a model designed for accurately quantifying the number of specific vehicles – particularly red cars – in given images. Utilizing the advanced capabilities of Azure OpenAI GPT-4 Turbo with Vision, this system meticulously analyzes each image, identifies and counts red cars, outputting the results in a clear JSON format. The process involves aggregating the counts from multiple images to provide a cumulative total, offering a precise and efficient approach to vehicle analysis.


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
        "image_1": image_1_path,
        "image_2": image_2_path,
        "system_message": "You are an AI assistant. You task is to check the number of red cars in the image. Reply in plain json, with key \"red cars\", and possible values of number of red cars."
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "response": "Total number of red cars: 3."
    }
}
```
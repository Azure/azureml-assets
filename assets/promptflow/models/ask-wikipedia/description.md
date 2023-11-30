This sample demonstrates the application of Q&A using GPT, enhanced by incorporating information from Wikipedia to make the answer more grounded. This process involves searching for a relevant Wikipedia link and extracting its page contents. The Wikipedia contents serve as an augmented prompt for the GPT chat API to generate a response.


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
        "question": "When did OpenAI announce GPT-4?"
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "answer": "OpenAI announced GPT-4 on March 14, 2023. [1] SOURCES: [1] https://en.wikipedia.org/w/index.php?search=GPT-4"
    }
}
```
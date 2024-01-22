The "Ask Wikipedia" is a Q&A model that employs GPT3.5 to answer questions using information sourced from Wikipedia, ensuring more grounded responses. This process involves identifying the relevant Wikipedia link and extracting its contents. These contents are then used as an augmented prompt, enabling GPT3.5 to generate an accurate response to the question.


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
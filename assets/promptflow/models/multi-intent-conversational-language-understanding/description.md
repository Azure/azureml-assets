The "Multi-Intent Conversational Language Understanding" is a standard model that utilizes Azure AI Language to extract intents from conversations. Azure AI language hosts its Conversational Language Understanding service (CLU), allowing users to analyze conversations and identify meaning. 


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
        "utterance": "Play BB King and increase the volume."
    }
}
```

#### Sample output
```json
{
    "outputs": {
      "intents": [
        {
          "query": "Play BB King.",
          "prediction": {
            "topIntent": "PlayMedia",
            "projectKind": "Conversation",
            "intents": [
              {
                "category": "PlayMedia",
                "confidenceScore": 0.99993074
              },
              {
                "category": "UpdateVolume",
                "confidenceScore": 0.00006925894
              },
              {
                "category": "None",
                "confidenceScore": 0
              }
            ],
            "entities": []
          }
        },
        {
          "query": "Increase the volume.",
          "prediction": {
            "topIntent": "UpdateVolume",
            "projectKind": "Conversation",
            "intents": [
              {
                "category": "UpdateVolume",
                "confidenceScore": 0.99994016
              },
              {
                "category": "PlayMedia",
                "confidenceScore": 0.000059839174
              },
              {
                "category": "None",
                "confidenceScore": 0
              }
            ],
            "entities": []
          }
        }
      ]
    }
}
```
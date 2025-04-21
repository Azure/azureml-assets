The "Analyze Conversations" is a standard model that utilizes Azure AI Language to perform various analyzes on text-based conversations. Azure AI language hosts pre-trained, task-oriented, and optimized conversation focused ML models, including various summarization aspects, PII entity extraction, etc. 


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
        "transcript_path": "<path_to_txt_file>"
    }
}
```

#### Sample output
Note: output has been shortened.
```json
{
    "outputs": {
      "narrative_summary": {
        "summaries": [
          {
            "aspect": "narrative",
            "text": "Ann Johnson, the host of \"Afternoon Cyber Tea\", welcomed the president of Microsoft Americas on the show. The president, who has a background in SAP and Standard Register, is known for her passion for building teams and developing individuals. She also serves as a board member for digital cloud and advisory services for Avanade and is an avid cyclist.",
            "contexts": [
              {
                "conversationItemId": "1",
                "offset": 0,
                "length": 962
              }
            ]
          }
        ],
        "id": "58",
        "warnings": []
      },
      "recap_summary": {
        "summaries": [
          {
            "aspect": "recap",
            "text": "The speaker, Generic, is the president of Microsoft Americas and has a long career in technology. She shares her experience and leadership philosophy, emphasizing the importance of team and accountability. She also encourages organizations to support aspiring women leaders by showing up as an ally and providing resources. Generic encourages everyone to advocate for women in technology and encourages them to share their knowledge and experience.",
            "contexts": [
              {
                "conversationItemId": "1",
                "offset": 0,
                "length": 962
              }
            ]
          }
        ],
        "id": "58",
        "warnings": []
      },
      "chapter_title": {
        "summaries": [
          {
            "aspect": "chapterTitle",
            "text": "Aftermath Cyber Tea: Exploring the Intersection of Innovation and Cybersecurity",
            "contexts": [
              {
                "conversationItemId": "1",
                "offset": 0,
                "length": 962
              },
              {
                "conversationItemId": "2",
                "offset": 0,
                "length": 334
              }
            ]
          }
        ],
        "id": "58",
        "warnings": []
      },
      "pii": {
        "id": "58",
        "conversationItems": [
          {
            "id": "1",
            "redactedContent": {
              "text": "Welcome to \"Afternoon Cyber Tea\" where we explore the intersection of innovation and cybersecurity. I am your host ***********. From the front lines of the digital defense to groundbreaking advancements shaping our digital future, we will bring you the latest insights, expert interviews, and captivating stories to stay one step ahead. [ Music ] Today we have a very special episode of \"Afternoon Cyber Tea\". I am thrilled, excited to be joined by ******** who is the president of Microsoft Americas. *** leads the $70 billion business responsible for delivering the full product and services portfolio of Microsoft to customers based in the United States, Canada, and Latin America. *** is a self-described \"team oriented\" leader with a passion for building teams and developing individuals. *** currently serves as a board member for digital cloud and advisory services for Avanade and serves as a board member for the famed luxury lifestyle leader ************."
            },
            "entities": [
              {
                "text": "Ann Johnson",
                "category": "Person",
                "offset": 115,
                "length": 11,
                "confidenceScore": 0.72
              }
            ]
          }
        ],
        "warnings": []
      },
      "key_phrases": {
        "id": "1",
        "keyPhrases": [
          "famed luxury lifestyle leader",
          "large German software company",
          "Afternoon Cyber Tea"
        ],
        "warnings": []
      }
    }
}
```
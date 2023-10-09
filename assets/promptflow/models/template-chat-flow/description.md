The "Template Chat Flow" is a chat model using GPT3.5 that generates the next message based on the conversation history and the latest chat content.


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
        "question": ["What is ChatGPT?"]
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "answer": "ChatGPT is a chatbot product developed by OpenAI. It is powered by the Generative Pre-trained Transformer (GPT) series of language models, with GPT-4 being the latest version. ChatGPT uses natural language processing to generate responses to user inputs in a conversational manner. It was released as ChatGPT Plus, a premium version, which provides enhanced features and access to the GPT-4 based version of OpenAI's API. ChatGPT allows users to interact and have conversations with the language model, utilizing both text and image inputs. It is designed to be more reliable, creative, and capable of handling nuanced instructions compared to previous versions. However, it is important to note that while GPT-4 improves upon its predecessors, it still retains some of the same limitations and challenges."
    }
}
```
# Template Chat Flow

This template provides the chat history support and a user-friendly chat interface in the authoring/debugging UI as prompt flow template.

## What you will learn

In this flow, you will learn
- how to compose a chat flow.
- prompt template format of LLM tool chat api. Message delimiter is a separate line containing role name and colon: "system:", "user:", "assistant:".
See <a href="https://platform.openai.com/docs/api-reference/chat/create#chat/create-role" target="_blank">OpenAI Chat</a> for more about message role.
    ```jinja
    system:
    You are a helpful assistant.

    user:
    {{question}}
    ```
- how to consume chat history in prompt.
    ```jinja
    {% for item in chat_history %}
    user:
    {{item.inputs.question}}
    assistant:
    {{item.outputs.answer}}
    {% endfor %}
    ```

## Getting started

### 1 Create connection for LLM tool to use
Go to "Prompt flow" "Connections" tab. Click on "Create" button, select one of LLM tool supported connection types and fill in the configurations.

Currently, there are two connection types supported by LLM tool: "AzureOpenAI" and "OpenAI". If you want to use "AzureOpenAI" connection type, you need to create an Azure OpenAI service first. Please refer to <a href="https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/" target="_blank">Azure OpenAI Service</a> for more details. If you want to use "OpenAI" connection type, you need to create an OpenAI account first. Please refer to <a href="https://platform.openai.com/" target="_blank">OpenAI</a> for more details.

### 2 Configure the flow with your connection
Click "Clone" button to start a new flow, and go to node "chat". Pick the connection you created in step 1 in the node parameter "Connection" dropdown list.

### 3 Start chatting
Click "Chat" button to open the chat window. Type in your question and click "Send" button. The chatbot will reply with an answer. You can continue chatting with the chatbot by typing in your next question and click "Send" button again.

## Used tools
- LLM Tool
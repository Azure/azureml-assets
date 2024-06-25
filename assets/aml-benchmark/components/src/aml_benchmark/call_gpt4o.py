import os

from openai import AzureOpenAI


endpoint = os.environ["AZURE_OPENAI_ENDPOINT_URL"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_version=api_version,
    api_key=api_key,
)
      
completion = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this picture:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://llava-vl.github.io/static/images/view.jpg"
                    }
                }
            ],
        },
    ],
)
      
print(completion.to_json())

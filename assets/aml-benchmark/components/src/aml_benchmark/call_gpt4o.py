import json
import os

from openai import AzureOpenAI, OpenAI


data = [
    {"prompt": "Describe this picture:", "image_url": "https://llava-vl.github.io/static/images/view.jpg"}
]

endpoint = os.environ["AZURE_OPENAI_ENDPOINT_URL"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_version=api_version,
    api_key=api_key,
)

"""
for d in data:
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
                        "text": d["prompt"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": d["image_url"]
                        }
                    }
                ],
            },
        ],
    )

    print(completion.to_json())
"""

batch_file_name = "batch.jsonl"
with open(batch_file_name, "wt") as f:
    for d in data:
        line = {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": deployment, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "text", "text": d["prompt"]}, {"type": "image_url", "image_url": {"url": d["image_url"]}}]}], "max_tokens": 10_000}}
        f.write(json.dumps(line))

batch_input_file = client.files.create(file=open("batch.jsonl", "rb"), purpose="batch")
# batch_input_file = client.files.create(file=open("batch.jsonl", "rb"), purpose="assistants")
# batch_input_file = client.files.create(file=open("batch.jsonl", "rb"), purpose="vision")

batch = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "do this quickly!"
    }
)
print(client.batches.retrieve(batch.id))

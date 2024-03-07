# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OpenAI utilities functions."""

from typing import Tuple, List

import os
import openai
import asyncio

import pkg_resources  # type: ignore[import]

openai_version_str = pkg_resources.get_distribution("openai").version
openai_version = pkg_resources.parse_version(openai_version_str)

if openai_version >= pkg_resources.parse_version("1.0.0"):
    _RETRY_ERRORS: Tuple = (
        openai.APIConnectionError,
        openai.APIError,
        openai.APIStatusError,
    )
else:
    _RETRY_ERRORS: Tuple = (  # type: ignore[no-redef]
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    )

_MAX_RETRIES = 7


def set_openai_api_parameters():
    """Set OpenAI parameter."""
    print(os.environ["OPENAI_API_TYPE"])
    print(os.environ["OPENAI_API_BASE"])
    print(os.environ["OPENAI_API_VERSION"])
    # print(os.environ["OPENAI_API_KEY"])
    openai.api_type = os.environ["OPENAI_API_TYPE"]
    openai.api_base = os.environ["OPENAI_API_BASE"]
    openai.api_version = os.environ["OPENAI_API_VERSION"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("OpenAI api variables are set.")


async def call_chat_completion_api_with_retries_async(
    messages: List,
    model="gpt4",
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
) -> str:
    """Call gpt chat completion api with retries."""
    n = 1
    while True:
        try:
            if openai_version >= pkg_resources.parse_version("1.0.0"):
                if openai.api_type.lower() == "azure":
                    from openai import AsyncAzureOpenAI

                    client = AsyncAzureOpenAI(
                        azure_endpoint=openai.api_base,
                        api_key=openai.api_key,
                        api_version=openai.api_version,
                        default_headers={"Client-User-Agent": "azureml/1.0"},
                    )
                    response = await client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    from openai import AsyncOpenAI

                    client = AsyncOpenAI(
                        api_key=openai.api_key,
                        default_headers={"Client-User-Agent": "azureml/1.0"},
                    )
                    response = await client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                return response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    engine=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                return response["choices"][0].message.content
        except _RETRY_ERRORS as e:
            if n > _MAX_RETRIES:
                raise
            secs = 2**n
            print(
                f"Retrying after {secs}s. API call failed due to {e.__class__.__name__}: {e}"
            )
            await asyncio.sleep(secs)
            n += 1
            continue

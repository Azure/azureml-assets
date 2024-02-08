# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for data distillation component."""

from abc import ABC, abstractmethod
from typing import List
from argparse import Namespace
import json
import asyncio

from openai_utils import (
    set_openai_api_parameters,
    call_chat_completion_api_with_retries_async,
)


class OpenAIDistillation(ABC):
    """Class for OAI Distillation."""

    def __init__(self, args: Namespace) -> None:
        """Init for OpenAIDistillation."""
        self.args = args
        set_openai_api_parameters()

    @abstractmethod
    def process_data(self, *args, **kwargs):
        """Process data."""
        pass

    def batch_process_data(self, raw_data: List) -> List:
        """Batch process data."""
        processed_data = []
        num_requests = len(raw_data)
        for idx, text in enumerate(raw_data):
            print(f"Processing request - {idx+1}/{num_requests}")
            new_data = self.process_data(text, idx)
            if len(new_data) == 0:
                print(
                    f"Could not get response from the OpenAI api, skipping record - {idx}."
                )
            else:
                processed_data.append(new_data)
                print(f"Got response from the OpenAI api, for record - {idx}.")

        return processed_data


class ZeroShotDistillation(OpenAIDistillation):
    """Class for OAI Zero Shot Distillation."""

    def process_data(self, text: str, idx: int) -> dict:
        """Process data."""
        processed_data = {}
        message_text = [
            {
                "role": "system",
                "content": "You are an AI assistant that summarizes text",
            },
            {
                "role": "user",
                "content": "Write a concise summary of the following: " + text,
            },
        ]
        try:
            prediction = asyncio.run(
                call_chat_completion_api_with_retries_async(
                    messages=message_text,
                    model="gpt-4",
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
            )
            print(prediction)
            processed_data = {
                "idx": idx,
                "text": text,
                "prediction": prediction,
            }
        except Exception as e:
            print(f"An error occurred: {e}")

        return processed_data


class ChainOfDensityDistillation(OpenAIDistillation):
    """Class for OAI Chain of Density Distillation."""

    def process_data(self, text: str, idx: int) -> dict:
        """Process data."""
        processed_data = {}

        # Chain of Density prompting
        cod_prompt = """
Article: {user_content}
You will generate increasingly concise, entity-dense summaries of the above article.
Repeat the following 2 steps {cod_steps} times.
Step 1. Identify 1-3 informative entities (";" delimited) from the article \
which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length \
which covers every entity and detail from the previous summary plus the missing entities.
A missing entity is:
- relevant to the main story,
- specific yet concise (5 words or fewer),
- novel (not in the previous summary),
- faithful (present in the article),
- anywhere (can be located anywhere in the article).
Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, \
containing little information beyond the entities marked as missing. \
Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Remember, use the exact same number of words for each summary.
Answer in JSON. The JSON should be a list (length 4) of dictionaries \
whose keys are "Missing_Entities" and "Denser_Summary".
        """

        user_content = text
        cod_steps = self.args.cod_steps

        summaryPrompt = cod_prompt.format(
            user_content=user_content, cod_steps=cod_steps
        )

        message_text = [
            {
                "role": "system",
                "content": "Write a concise, entity-dense summary of the following article.",
            },
            {"role": "user", "content": summaryPrompt},
        ]

        try:
            prediction = asyncio.run(
                call_chat_completion_api_with_retries_async(
                    messages=message_text,
                    model="gpt-4",  # cod
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
            )
            try:
                summaries = json.loads(prediction)
                last_summary = summaries[-1]["Denser_Summary"]
                print(f"user_content:\n{user_content}")
                print(f"last_summary:\n{last_summary}")
                processed_data = {
                    "idx": idx,
                    "text": user_content,
                    "prediction": last_summary,
                }
            except Exception as e:  # Log exception if JSON parsing fails
                print("Error", idx + 1, e)

        except Exception as e:
            print("Error", idx + 1)
            print(e)

        return processed_data

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PromptFactory class to create prompts from data."""


import json
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from jinja2 import Environment

from azureml._common._error_definition.azureml_error import AzureMLError

from aml_benchmark.utils.exceptions import BenchmarkValidationException
from aml_benchmark.utils.error_definitions import BenchmarkValidationError
from aml_benchmark.utils.exceptions import BenchmarkUserException
from aml_benchmark.utils.error_definitions import BenchmarkUserError
from .prompt import (
    PromptType,
    Prompt,
    CompletionsPrompt,
    ChatPrompt,
    OpenAICreate,
    OpenAICreateChatPrompt,
    Role,
)


logger = logging.getLogger(__name__)

PATIENCE = 100

JINJA_ENV = Environment(keep_trailing_newline=True)


@dataclass
class PromptFactory(ABC):
    """Factory for creating prompts."""

    n_shots: int
    prompt_pattern: str
    few_shot_pool: Optional[List] = None
    few_shot_pattern: Optional[str] = None
    few_shot_separator: Optional[str] = None
    prefix: Optional[str] = None
    ground_truth_column_name: Optional[str] = None
    additional_columns: Optional[str] = None
    label_map_str: Optional[str] = None
    output_pattern: Optional[str] = None
    system_message: Optional[str] = None
    metadata_keys: Optional[str] = None
    additional_payload: Optional[str] = None
    _messages: Optional[OpenAICreate] = None

    def __post_init__(self):
        """Validate and initialize the prompt factory."""
        if self.n_shots > 0 and self.n_shots > len(self.few_shot_pool) and self.few_shot_pattern:
            mssg = f"n_shots ({self.n_shots}) > |few_shot pool| ({len(self.few_shot_pool)})"
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg))

        self.label_map_jinja_prefix = ""
        if self.label_map_str:
            self.label_map_jinja_prefix = self._create_label_map_jinja_prefix(self.label_map_str)
            self._inverse_label_map = self._inverse_label_map(self.label_map_str)

        # Validate prompt_pattern
        if self.prompt_pattern:
            self.prompt_pattern = self.label_map_jinja_prefix + self.prompt_pattern
            self.validate_jinja_template(self.prompt_pattern, "Prompt pattern is not a valid jinja pattern.")
        else:
            mssg = "A prompt pattern (in jinja template) is required."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg))

        # Validate output_pattern
        if self.output_pattern:
            self.validate_jinja_template(self.output_pattern,
                                         "Output pattern is not a valid jinja pattern.")

        # Validate few_shot_pattern
        if self.n_shots > 0:
            if self.few_shot_pattern:
                self.augmented_few_shot_pattern = self.label_map_jinja_prefix + self.few_shot_pattern
            else:
                self.augmented_few_shot_pattern = self.prompt_pattern + self.output_pattern
            self.validate_jinja_template(self.augmented_few_shot_pattern,
                                         "Few shot pattern is not a valid jinja pattern.")

    def validate_jinja_template(self, template: str, error_message: str):
        """Validate a jinja template."""
        try:
            _ = JINJA_ENV.from_string(template)
        except Exception:
            mssg = f"{template} is not a valid jinja pattern. Error: {error_message}"
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg))

    @classmethod
    def from_type(cls, prompt_type: PromptType):
        """Create a prompt factory from a prompt type."""
        if prompt_type == PromptType.completions.name:
            return CompletionsPromptFactory
        elif prompt_type == PromptType.chat.name:
            return ChatPromptFactory
        else:
            mssg = f"Unrecognized prompt type {prompt_type}. Should be \
                             one of {PromptType.chat.name} or {PromptType.completions.name}"
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg))

    @abstractmethod
    def create_prompt(row: Dict) -> Prompt:
        """Create prompts from a row of data."""
        pass

    @staticmethod
    def _parse_label_map(label_map_str: str) -> Dict:
        """Parse label map string into a dictionary."""
        try:
            label_map = json.loads(label_map_str)
            label_map = {int(k): v for k, v in label_map.items()}
        except Exception:
            mssg = f"Invalid label map string:\n{label_map_str}"
            logger.exception(mssg)
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg))
        return label_map

    @staticmethod
    def _create_label_map_jinja_prefix(label_map_str: str) -> str:
        label_map = PromptFactory._parse_label_map(label_map_str)
        label_map_jinja_prefix = f"{{% set label_map = {label_map} %}}"
        logger.info(f"Created label map jinja prefix:\n{label_map_jinja_prefix}")
        try:
            _ = JINJA_ENV.from_string(label_map_jinja_prefix)
        except Exception:
            mssg = "Label map is not a valid jinja template."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg))
        return label_map_jinja_prefix

    @staticmethod
    def _inverse_label_map(label_map_str: str) -> Dict:
        label_map = PromptFactory._parse_label_map(label_map_str)
        return {value: key for key, value in label_map.items()}

    def _sample_few_shots(self, row: Dict[str, str]):
        few_shots = []
        few_shot_ids = set()

        retries = 0
        while len(few_shots) < self.n_shots:

            retries += 1
            if retries > PATIENCE:
                mssg = f"Unable to find {self.n_shots} few shots after {PATIENCE} retries"
                raise BenchmarkValidationException._with_error(
                    AzureMLError.create(BenchmarkValidationError, error_details=mssg))

            id_ = random.randint(0, len(self.few_shot_pool) - 1)
            candidate = self.few_shot_pool[id_]

            if id_ in few_shot_ids or all(row.get(k, "") == candidate.get(k, "") for k in row):
                continue

            few_shots.append(candidate)
            few_shot_ids.add(id_)

        return few_shots

    def get_label_from_output_pattern(self, row) -> str:
        """Get label from output jinja pattern."""
        if self.label_map_str:
            output_label_map_pattern = self.label_map_jinja_prefix + self.output_pattern
            output_label = JINJA_ENV.from_string(output_label_map_pattern).render(row)
            if output_label in self._inverse_label_map:
                return self._inverse_label_map[output_label]
            else:
                return output_label
        else:  # parse the label direcly
            label_template = JINJA_ENV.from_string(self.output_pattern)
            return label_template.render(row)

    def process_row(self, row: Dict, index: int) -> Dict:
        """Process a row of data and return the output data."""
        prompt = self.create_prompt(row=row)
        output_data = prompt.to_openai_create_prompt()
        output_data["prompt_length"] = len(prompt)

        if self.output_pattern is not None:
            output_data['completion'] = self.get_label_from_output_pattern(row)

        if self.ground_truth_column_name is not None and len(self.ground_truth_column_name) > 0:
            if self.ground_truth_column_name in row:
                output_data['ground_truth'] = row[self.ground_truth_column_name]
            else:
                mssg = "Ground truth column is not present in the data"
                raise BenchmarkValidationException._with_error(
                    AzureMLError.create(BenchmarkValidationError, error_details=mssg))

        if self.additional_columns:
            elements = self.additional_columns.split(",")
            strips = [s.strip() for s in elements if s.strip()]
            for k in strips:
                try:
                    output_data[k] = row[k]
                except KeyError:
                    raise BenchmarkUserException._with_error(
                        AzureMLError.create(
                            BenchmarkUserError,
                            error_details=f"Column {k} doesn't exist. Please check your data before submitting again.")
                        )

        if self.metadata_keys is not None:
            for k in self.metadata_keys.split(","):
                k = k.strip()
                if k in row:
                    output_data[k] = row[k]
                else:
                    logger.warning(f"Metadata key {k} not found in data at row {index}")

        if self.additional_payload:
            try:
                additional_payload = json.loads(self.additional_payload)
            except Exception:
                mssg = "Additional payload is not a valid json"
                raise BenchmarkValidationException._with_error(
                    AzureMLError.create(BenchmarkValidationError, error_details=mssg))
        else:
            additional_payload = {}

        output_data = {
            **output_data,
            **additional_payload,
        }

        return output_data


@dataclass
class CompletionsPromptFactory(PromptFactory):
    """Factory for completions prompts."""

    def create_prompt(self, row: Dict) -> CompletionsPrompt:
        """Create completions prompts."""
        prompt_template = JINJA_ENV.from_string(self.prompt_pattern)
        prompt = str(prompt_template.render(row))

        if self.few_shot_pool is not None and self.n_shots > 0:
            few_shots = self._create_few_shots(row)
            prompt = few_shots + prompt

        # add prefix
        if self.prefix is not None:
            prompt = self.prefix + prompt

        prompt = CompletionsPrompt(prompt)

        return prompt

    def _create_few_shots(self, row: Dict[str, str]) -> OpenAICreateChatPrompt:
        """Create few shot prompts for completions prompt."""
        few_shots = self._sample_few_shots(row)

        few_shot_prompt = ""
        for few_shot in few_shots:
            # sometimes few_shot_pattern is empty, while n_shots > 0
            # this is useful for some configurations, e.g. chain-of-thought, where demonstrations
            # are precrafted in e.g. prefix, but we still want to use n_shots
            if self.augmented_few_shot_pattern:
                few_shot_template = JINJA_ENV.from_string(self.augmented_few_shot_pattern)
                few_shot_prompt += few_shot_template.render(few_shot)

            if self.few_shot_separator is not None:
                few_shot_prompt += self.few_shot_separator

        return few_shot_prompt


@dataclass
class ChatPromptFactory(PromptFactory):
    """Factory for chat prompts."""

    def create_prompt(self, row: Dict) -> ChatPrompt:
        """Create chat prompts."""
        messages = []

        if self.system_message:
            system_message = self._parse_affix(self.system_message,
                                               Role.system.name)
            messages.extend(system_message)

        prefix_messages = []
        if self.prefix:
            prefix_messages = self._parse_affix(affix=self.prefix)

        # Parsing the prompt pattern
        input_messages = self._parse_affix(affix=self.prompt_pattern)

        if self.few_shot_pool is not None and self.n_shots > 0:
            few_shot_messages = self._create_few_shots(row)
            if len(prefix_messages) == 1 and len(few_shot_messages) > 0:
                # Updating the first few shot message with prefix
                few_shot_messages[0]["content"] = prefix_messages[0]["content"] + few_shot_messages[0]["content"]
            elif len(prefix_messages) > 1:
                # If prefix is a list of dictionaries
                messages.extend(prefix_messages)

            messages.extend(few_shot_messages)
        else:
            # With no few shot data and prefix present, adding prefix to prompt
            if len(prefix_messages) == 1:
                # Updating the first few shot message with prefix
                input_messages[0]["content"] = prefix_messages[0]["content"] + input_messages[0]["content"]
            elif len(prefix_messages) > 1:
                # If prefix is a list of dictionaries
                input_messages = prefix_messages + input_messages

        for input_message in input_messages:
            input_template = JINJA_ENV.from_string(input_message["content"])
            input_message["content"] = str(input_template.render(row))

        messages.extend(input_messages)
        prompt = ChatPrompt(messages)

        return prompt

    def _create_few_shots(self, row: Dict[str, str]) -> OpenAICreateChatPrompt:
        few_shots = self._sample_few_shots(row)

        few_shot_messages = []
        if self.few_shot_pattern:
            # Different chat prompt style when user provides few_shot_pattern
            few_shot_template = JINJA_ENV.from_string(self.augmented_few_shot_pattern)
            for few_shot_dict in few_shots:
                messages = str(few_shot_template.render(few_shot_dict))
                messages = self._parse_affix(affix=messages)
                few_shot_messages.extend(messages)
        else:
            user_template = JINJA_ENV.from_string(self.prompt_pattern)
            assistant_template = JINJA_ENV.from_string(self.output_pattern)
            for few_shot_dict in few_shots:
                # User role for few shot chat
                messages = str(user_template.render(few_shot_dict))
                messages = self._parse_affix(messages, Role.user.name)
                few_shot_messages.extend(messages)
                # Assistant response for few shot chat
                messages = str(assistant_template.render(few_shot_dict))
                messages = self._parse_affix(messages, Role.assistant.name)
                few_shot_messages.extend(messages)

        return few_shot_messages

    def _parse_affix(self, affix: str, role="user") -> OpenAICreateChatPrompt:
        """Convert affix string to messages format."""
        try:
            messages = json.loads(affix)
        except json.decoder.JSONDecodeError:
            # NOTE: user role is currently prefered over system role
            # though this may change in the future as models are updated
            messages = [{"role": role, "content": affix}]

        # Handling the case where Json.loads load a string with only number as intger.
        if isinstance(messages, int):
            messages = [{"role": role, "content": affix}]

        messages = self._make_list(messages)
        self._validate_messages(messages)
        return messages

    @staticmethod
    def _validate_messages(messages: OpenAICreateChatPrompt):
        for msg in messages:
            assert "role" in msg, f"Missing role in message: {msg}"
            assert "content" in msg, f"Missing content in message: {msg}"

    @staticmethod
    def _make_list(messages: Union[Dict, List]):
        if isinstance(messages, dict):
            messages = [messages]
        return messages

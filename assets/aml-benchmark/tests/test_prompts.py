# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the functionality of the prompt factory which powers the prompt crafter."""

import sys
import pytest
from typing import List, Dict

from .test_utils import get_src_dir

sys.path.append(get_src_dir())
try:
    from prompt_crafter.package.prompt_factory import ChatPromptFactory, CompletionsPromptFactory
except ImportError:
    raise ImportError("Please install the package 'prompt_crafter' to run this test.")


# Without few_shot_pattern as inputs
expected_op_str_response = [
                    {"role": "system", "content": "You are an assistant."},
                    {"role": "user", "content": "fs_in?"},
                    {"role": "assistant", "content": "fs_out"},
                    {"role": "user", "content": "in4?"}
                ]

expected_op_int_response = [
            {"role": "user", "content": "fs_in?"},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": "in4?"}
        ]

system_message = "You are an assistant."


@pytest.mark.parametrize(
    "n_shots, few_shot_pool, row, expected_output, system_message",
    [
        (1, [{'input': 'fs_in', 'output': 'fs_out'}],
         {'input': 'in4'},
         expected_op_str_response, system_message),
        (1, [{'input': 'fs_in', 'output': '1'}],
         {'input': 'in4'},
         expected_op_int_response, None),
        (0, [{'input': 'fs_in', 'output': 'fs_out'}],
         {'input': 'in4'},
         [{"role": "user", "content": "in4?"}], None),
    ]
)
def test_chat_prompts(
    n_shots: int,
    few_shot_pool: List[Dict[str, str]],
    row: Dict[str, str],
    expected_output: str,
    system_message: str,
) -> None:
    """Test the functionality of the chat prompts with various inputs.

    The role of the few shot prompts outputs should be correctly set to user/
    assistant roles. If few_shot_pattern is present, then assistant role is
    not added to the few_shot_prompts.
    """
    prompt_pattern = "{{input}}?"
    output_pattern = '{{output}}'
    few_shot_separator = "\n"

    prompt_factory = ChatPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        output_pattern=output_pattern,
        few_shot_separator=few_shot_separator,
        system_message=system_message)
    prompt = prompt_factory.create_prompt(row=row)
    assert prompt.raw_prompt == expected_output


def test_chat_prompts_with_few_shot_pattern() -> None:
    """Test the functionality of the chat prompts with few_shot_pattern."""
    n_shots = 1
    prompt_pattern = "{{input}}?"
    output_pattern = "{{output}}"
    few_shot_separator = "\n"
    few_shot_pool = [{'input': 'fs_in', 'output': 'fs_out'}]
    row = {'input': 'in'}
    # When few_shot_prompt is present in input
    few_shot_pattern = '[{"role": "user", "content": "{{input}}?{{output}}"}]'
    expected_output_with_few_shot_pattern = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "fs_in?fs_out"},
        {"role": "user", "content": "in?"}]
    prompt_factory = ChatPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        output_pattern=output_pattern,
        few_shot_separator=few_shot_separator,
        system_message=system_message,
        few_shot_pattern=few_shot_pattern)
    prompt = prompt_factory.create_prompt(row=row)
    assert prompt.raw_prompt == expected_output_with_few_shot_pattern

    # Without system message test
    expected_output_with_few_shot_pattern = [
        {"role": "user", "content": "fs_in?fs_out"},
        {"role": "user", "content": "in?"}]
    prompt_factory = ChatPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        output_pattern=output_pattern,
        few_shot_separator=few_shot_separator,
        system_message=None,
        few_shot_pattern=few_shot_pattern)
    prompt = prompt_factory.create_prompt(row=row)
    assert prompt.raw_prompt == expected_output_with_few_shot_pattern


@pytest.mark.parametrize(
    "n_shots, few_shot_pool, row, expected_output",
    [
        (1, [{'input': 'in', 'output': 'out'}],
         {'input': 'in4', 'output': 'out4'},
         'Input:in\nOutput:out\nInput:in4\nOutput:'),
        (1, [{'input': 'in', 'output': '1'}],
         {'input': 'in4', 'output': '2'},
         'Input:in\nOutput:1\nInput:in4\nOutput:'),
        (0, [{'input': 'in', 'output': 'out'}],
         {'input': 'in4', 'output': 'out4'},
         'Input:in4\nOutput:'),
        (0, [{'input': 'in', 'output': '1'}],
         {'input': 'in4', 'output': '2'},
         'Input:in4\nOutput:')
    ]
)
def test_completions_prompts(
    n_shots: int,
    few_shot_pool: List[Dict[str, str]],
    row: Dict[str, str],
    expected_output: str
) -> None:
    """Test the functionality of the completions prompt factory."""
    prompt_pattern = 'Input:{{input}}\nOutput:'
    output_pattern = '{{output}}'
    few_shot_pattern = 'Input:{{input}}\nOutput:{{output}}'
    few_shot_separator = "\n"
    prefix = None
    system_message = "You are an AI assistant."

    # When few_shot_pattern is present
    prompt_factory = CompletionsPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_pattern=few_shot_pattern,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        system_message=system_message,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)

    assert prompt.raw_prompt == expected_output

    # When few_shot_pattern is absent, it will be created
    # from prompt_pattern and output_pattern
    prompt_factory = CompletionsPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        system_message=system_message,
        output_pattern=output_pattern,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)

    assert prompt.raw_prompt == expected_output

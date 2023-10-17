# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the functionality of the prompt factory which powers the prompt crafter."""

import sys
from .test_utils import get_src_dir

sys.path.append(get_src_dir())
try:
    from prompt_crafter.package.prompt_factory import ChatPromptFactory, CompletionsPromptFactory
except ImportError:
    raise ImportError("Please install the package 'prompt_crafter' to run this test.")


def test_chat_prompts() -> None:
    """Test the functionality of the chat prompts with various inputs.

    The role of the few shot prompts outputs should be correctly set to user/
    assistant roles. If few_shot_pattern is present, then assistant role is
    not added to the few_shot_prompts.
    """
    n_shots = 1
    prompt_pattern = "{{input}}?"
    output_pattern = '{{output}}'
    few_shot_pool = [{'input': 'fs_in', 'output': 'fs_out'}]
    few_shot_separator = "\n"
    system_message = "You are an assistant."
    row = {'input': 'in4'}

    # Without few_shot_pattern as inputs
    expected_output = [{"role": "system", "content": "You are an assistant."},
                       {"role": "user", "content": "fs_in?"},
                       {"role": "assistant", "content": "fs_out"},
                       {"role": "user", "content": "in4?"}]
    prompt_factory = ChatPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        output_pattern=output_pattern,
        few_shot_separator=few_shot_separator,
        system_message=system_message)
    prompt = prompt_factory.create_prompt(row=row)
    assert prompt.raw_prompt == expected_output

    # When few_shot_prompt is present in input
    few_shot_pattern = '[{"role": "user", "content": "{{input}}?{{output}}"}]'
    expected_output_with_few_shot_pattern = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "fs_in?fs_out"},
        {"role": "user", "content": "in4?"}]
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


def test_completions_prompts() -> None:
    """Test the functionality of the completions prompt factory."""
    n_shots = 1
    prompt_pattern = 'Input:{{input}}\nOutput:'
    few_shot_pattern = 'Input:{{input}}\nOutput:{{output}}'
    few_shot_pool = [{'input': 'in', 'output': 'out'}]
    few_shot_separator = "\n"
    prefix = None
    row = {'input': 'in4', 'output': 'out4'}
    system_message = "You are an AI assistant."
    expected_output = 'Input:in\nOutput:out\nInput:in4\nOutput:'
    output_pattern = '{{output}}'

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

    # When few_shot_pattern is absent, it will be created from prompt_pattern and output_pattern
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

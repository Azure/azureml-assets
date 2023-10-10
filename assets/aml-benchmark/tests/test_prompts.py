import os
import sys
from test_utils import get_src_dir


PROMPT_CRAFTER_NAME = "prompt_crafter"
sys.path.insert(0, os.path.join(get_src_dir(), PROMPT_CRAFTER_NAME))
from package_3p.prompt_factory import ChatPromptFactory, CompletionsPromptFactory


def test_chat_prompts():
    """Test that the role of the few shot prompts outputs are correctly set to user/assistant 
    roles. This is applicable when few_shot_prompts are not present in the input data.
    If few_shot_pattern is present, then assistant role is not added to the few_shot_prompts."""
    n_shots = 1
    prompt_pattern = '[{"role": "user", "content": "{{input}}?"}]'
    output_pattern = '{{output}}'
    few_shot_pool = [{'input': 'fs_in', 'output': 'fs_out'}]
    few_shot_separator = "\n"
    system_message = "You are an assistant."
    row = {'input': 'in4'}
    expected_output = [{"role": "system", "content": "You are an assistant."},
                       {"role": "user", "content": "fs_in?"},
                       {"role": "assistant", "content": "fs_out"},
                       {"role": "user", "content": "in4?"}]

    few_shot_pattern = '[{"role": "user", "content": "{{input}}?{{output}}"}]'
    expected_output_with_few_shot_pattern = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "fs_in?fs_out"},
        {"role": "user", "content": "in4?"}]

    # Without few_shot_pattern as inputs
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


def test_completions_prompts():
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
        few_shot_pattern="",
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        system_message=system_message,
        output_pattern=output_pattern,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)

    assert prompt.raw_prompt == expected_output

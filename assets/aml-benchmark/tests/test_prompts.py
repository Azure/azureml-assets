import os
import sys
import pytest

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_path)  

from package_3p.prompt_factory import ChatPromptFactory, CompletionsPromptFactory
from prompt_crafter import GPTVPromptFactory

@pytest.mark.skip("TODO: Fix this test")
def test_chat_standard():
    n_shots = 1
    prompt_pattern = '[{"role": "user", "content": "{input}"}]'
    few_shot_pattern = '[{"role": "user", "content": "{input}"}, {"role": "system", "content": "{output}"}]'
    few_shot_pool = [{'input': 'in', 'output': 'out'}]
    few_shot_separator = "\n"
    prefix = None
    row={'input': 'in4'}
    expected_output = [{"role": "user", "content": "in"},{"role": "system", "content": "out"},{"role": "user", "content": "in4"}]

    prompt_factory = ChatPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_pattern=few_shot_pattern,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)

    assert prompt.raw_prompt == expected_output

@pytest.mark.skip("TODO: Fix this test")
def test_completions_standard():
    n_shots = 1
    prompt_pattern = 'Input: "{input}"'
    few_shot_pattern = 'Input: "{input}"\nOutput: "{output}"'
    few_shot_pool = [{'input': 'in', 'output': 'out'}]
    few_shot_separator = "\n"
    prefix = None
    row={'input': 'in4'}
    expected_output = 'Input: "in"\nOutput: "out"\nInput: "in4"'

    prompt_factory = CompletionsPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_pattern=few_shot_pattern,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)
    
    assert prompt.raw_prompt == expected_output

def test_gptv_standard():
    prompt_pattern = '[{"role": "system", "content": "hello"}, \
        {"role": "user", "content": ["{{text}}", {"image": "{{image}}"}]}]'
    row={'image': '/xsdf', 'text': "test"}
    expected_output = [{"role": "system", "content": "hello"}, \
                       {"role": "user", "content": ["test", {"image": "/xsdf"}]}]

    prompt_factory = GPTVPromptFactory(
        n_shots=0,
        prompt_pattern=prompt_pattern,
        )

    prompt = prompt_factory.create_prompt(row=row)
    
    assert prompt.raw_prompt == expected_output

def test_gptv_multi_image_input():
    n_shots = 0
    prompt_pattern = '[{"role": "system", "content": [{"image1": "{{image1}}"}, {"image2": "{{image2}}"}]}]'
    row={"image1": "cdfvsa", "image2": "xasfvr"}
    expected_output = [{"role": "system", "content": [{"image1": "cdfvsa"}, {"image2": "xasfvr"}]}]

    prompt_factory = GPTVPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
    )

    prompt = prompt_factory.create_prompt(row=row)
    
    assert prompt.raw_prompt == expected_output

def test_gptv_multi_text_input():
    n_shots = 0
    prompt_pattern = '[{"role": "user", "content": ["{{text1}}", "{{text2}}", "{{text3}}"]}]'
    row={"text1": "how", "text2": "are", "text3": "you"}
    expected_output = [{"role": "user", "content": ["how", "are", "you"]}]

    prompt_factory = GPTVPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
    )

    prompt = prompt_factory.create_prompt(row=row)
    
    assert prompt.raw_prompt == expected_output

@pytest.mark.skip("TODO: Fix this test")
def test_chat_formatted_input():
    n_shots = 0
    prompt_pattern = ''
    few_shot_pattern = ''
    few_shot_separator = "\n"
    few_shot_pool = []
    prefix = None

    row={"input": [{"role": "user", "content": "in"},{"role": "system", "content": "out"},{"role": "user", "content": "in4"}]}
    expected_output = [{"role": "user", "content": "in"},{"role": "system", "content": "out"},{"role": "user", "content": "in4"}]

    prompt_factory = ChatPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pattern=few_shot_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)
    
    assert prompt.raw_prompt == expected_output

@pytest.mark.skip("TODO: Fix this test")
def test_completions_formatted_input():
    n_shots = 0
    prompt_pattern = ''
    few_shot_pattern = ''
    few_shot_pool = []
    few_shot_separator = "\n"
    prefix = None
    row={"input": 'Input: "in"\nOutput: "out"\nInput: "in4"'}
    expected_output = 'Input: "in"\nOutput: "out"\nInput: "in4"'

    prompt_factory = CompletionsPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_pattern=few_shot_pattern,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)
    
    assert prompt.raw_prompt == expected_output

@pytest.mark.skip("TODO: Fix this test")
def test_chat_prefix():
    n_shots = 0
    prompt_pattern = ''
    few_shot_pattern = ''
    few_shot_separator = "\n"
    few_shot_pool = []
    prefix = '[{"role": "user", "content": "in"},{"role": "system", "content": "out"}]'
    row={"input": [{"role": "user", "content": "in4"}]}
    expected_output = [{"role": "user", "content": "in"},{"role": "system", "content": "out"},{"role": "user", "content": "in4"}]

    prompt_factory = ChatPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pattern=few_shot_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)

    print(prompt.raw_prompt)
    assert prompt.raw_prompt == expected_output

@pytest.mark.skip("TODO: Fix this test")
def test_completions_prefix():
    n_shots = 0
    prompt_pattern = ''
    few_shot_pattern = ''
    few_shot_pool = []
    few_shot_separator = "\n"
    prefix = '{{Input}}: "in"\n{{Output}}: "out"\n'
    row={"input": 'Input: "in4"'}
    expected_output = 'Input: "in"\nOutput: "out"\nInput: "in4"'

    prompt_factory = CompletionsPromptFactory(
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        few_shot_pool=few_shot_pool,
        few_shot_pattern=few_shot_pattern,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        label_map_str=None)

    prompt = prompt_factory.create_prompt(row=row)
    
    assert prompt.raw_prompt == expected_output


def test_chat_roles():
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
 

def test_completions_system_message():
    n_shots = 1
    prompt_pattern = 'Input:{{input}}\nOutput:'
    few_shot_pattern = 'Input:{{input}}\nOutput:{{output}}'
    few_shot_pool = [{'input': 'in', 'output': 'out'}]
    few_shot_separator = "\n"
    prefix = None
    row={'input': 'in4','output':'out4'}
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

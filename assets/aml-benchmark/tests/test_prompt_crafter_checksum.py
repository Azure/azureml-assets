# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Prompt Crafter Checksums for various samples."""

import os
import sys
import pytest
import mlflow

from .test_utils import get_src_dir

sys.path.append(get_src_dir())
try:
    from aml_benchmark.prompt_crafter.package.prompt_crafter import PromptCrafter
except ImportError:
    raise ImportError("Please install the package 'prompt_crafter' to run this test.")


# Setting dataset path
TEST_DATA = os.path.join(
            os.path.dirname(__file__), 'data/test_data_prompt_crafter')
OUTPUT_PATH = os.path.join(
            os.path.dirname(__file__), 'data')


def setup_folder(dataset_name: str):
    """Set the dataset paths."""
    dataset_path = os.path.join(TEST_DATA, dataset_name)
    test_output_path = os.path.join(OUTPUT_PATH, dataset_name, "output_dir")
    test_output_mltable_path = os.path.join(OUTPUT_PATH, dataset_name, "output_mltable")

    os.makedirs(test_output_path, exist_ok=True)
    os.makedirs(test_output_mltable_path, exist_ok=True)
    return dataset_path, test_output_path, test_output_mltable_path


def base_test_output(prompt_crafter: PromptCrafter,
                     output_dir_path: str,
                     output_mltable_path: str) -> None:
    """Check if the output files exist."""
    assert os.path.exists(os.path.join(output_dir_path, prompt_crafter.OUTPUT_FILENAME))
    assert os.path.exists(os.path.join(output_mltable_path, prompt_crafter.OUTPUT_FILENAME))
    assert os.path.exists(os.path.join(output_mltable_path, prompt_crafter.MLTABLE_FILENAME))


def base_test(dataset_name,
              completion_ground_truth_checksum,
              chat_ground_truth_checksum,
              params) -> None:
    """Check output checksums for completion and chat prompt types."""
    dataset_path, test_output_path, test_output_mltable_path = setup_folder(dataset_name)

    if completion_ground_truth_checksum:
        with mlflow.start_run():
            params["prompt_type"] = "completions"
            prompt_crafter = PromptCrafter(
                test_data=os.path.join(dataset_path, "inference_sample.jsonl"),
                few_shot_data=os.path.join(dataset_path, "fewshot_sample.jsonl"),
                output_file=os.path.join(test_output_path, PromptCrafter.OUTPUT_FILENAME),
                output_mltable=test_output_mltable_path,
                **params
            )
            checksum = prompt_crafter.run()
            assert checksum == completion_ground_truth_checksum
            base_test_output(
                prompt_crafter=prompt_crafter,
                output_dir_path=test_output_path,
                output_mltable_path=test_output_mltable_path
            )

    if chat_ground_truth_checksum:
        with mlflow.start_run():
            params["prompt_type"] = "chat"
            prompt_crafter = PromptCrafter(
                test_data=os.path.join(dataset_path, "inference_sample.jsonl"),
                few_shot_data=os.path.join(dataset_path, "fewshot_sample.jsonl"),
                output_file=os.path.join(test_output_path, PromptCrafter.OUTPUT_FILENAME),
                output_mltable=test_output_mltable_path,
                **params
            )
            checksum = prompt_crafter.run()
            assert checksum == chat_ground_truth_checksum


# Test cases
@pytest.mark.test_mnli
def test_mnli() -> None:
    """Test to ensure that the checksums for the mnli dataset are as expected."""
    completion_ground_truth_checksum = "25ed56454fc4f7d5c042372e34a32f0a14ec7b5779deec0ab57a6be82aaee258"
    chat_ground_truth_checksum = "541038f3df8cb820729f633a9149c0c3a13daa81bf2a9ab2c6be3a3501e67942"
    params = {
        "n_shots": 7,
        "random_seed": 0,
        "metadata_keys": "label",
        "prompt_pattern": "### Premise:\n\n{{sentence1}}\n\n### Hypothesis:\n\n{{sentence2}}\n\n### Possible labels:\n\nentailment, neutral, contradiction\n\n### Label:",  # noqa: E501
        "few_shot_pattern": "### Premise:\n\n{{sentence1}}\n\n### Hypothesis:\n\n{{sentence2}}\n\n### Possible labels:\n\nentailment, neutral, contradiction\n\n### Label:\n\n{{label_map[label]}}\n\n",  # noqa: E501
        "few_shot_separator": None,
        "prefix": None,
        "label_map": '{"0":"entailment","1":"neutral","2":"contradiction"}',
        "additional_payload": None,
        "output_pattern": "{{label}}",
        "system_message": None,
    }
    base_test("mnli", completion_ground_truth_checksum, chat_ground_truth_checksum, params)


@pytest.mark.test_anli
def test_anli() -> None:
    """Test to ensure that the checksums for the anli dataset are as expected."""
    completion_ground_truth_checksum = "1e614c621f02f318c55877ac41e8244f9659ea4902971d5ae5a5ef038071539c"
    chat_ground_truth_checksum = "5e1ba228d89613cedc2f523922da582dbc3c02c5e2e6cab85964c1bab3835fd3"
    params = {
        "n_shots": 7,
        "random_seed": 0,
        "output_pattern": "{{label}}",
        "system_message": None,
        "metadata_keys": "label",
        "prompt_pattern": "Given the following context:\n\n{{premise}}\n\nQuestion:{{hypothesis}}\n\nTrue, False or Neither?\n\nThe answer is:",  # noqa: E501
        "few_shot_pattern": "Given the following context:\n\n{{premise}}\n\nQuestion:{{hypothesis}}\n\nTrue, False or Neither?\n\nThe answer is: {{label_map[label]}}\n\n",  # noqa: E501
        "few_shot_separator": None,
        "prefix": None,
        "label_map": '{"0":"True","1":"Neither","2":"False"}',
        "additional_payload": None,
    }
    base_test("anli", completion_ground_truth_checksum, chat_ground_truth_checksum, params)


@pytest.mark.test_gsm8k_chain_of_thought
def test_gsm8k_chain_of_thought() -> None:
    """Test to ensure that the checksums for the gsm8k_chain_of_thought dataset are as expected."""
    completion_ground_truth_checksum = "a087a01eb5cf98c1700f9aa4415ebb920418d6b0311998955e29dff757aef712"
    chat_ground_truth_checksum = "6b03925c3bcf5923182f7a9bb9e28ade7754ad21d9bb78d85ab1793e384bb2ff"
    params = {
        "n_shots": 0,
        "random_seed": 0,
        "output_pattern": "{{answer}}",
        "system_message": None,
        "metadata_keys": "answer",
        "prompt_pattern": "{{prompt}}",
        "few_shot_pattern": None,
        "few_shot_separator": None,
        "prefix": None,
        "label_map": None,
        "additional_payload": None,
    }
    base_test("gsm8k_chain_of_thought", completion_ground_truth_checksum, chat_ground_truth_checksum, params)


@pytest.mark.test_bigbench_hard_casual_judgment
def test_bigbench_hard_casual_judgment() -> None:
    """Test to ensure that the checksums for the bigbench-hard_causal_judgement dataset are as expected."""
    completion_ground_truth_checksum = None
    chat_ground_truth_checksum = "68a53edfe173c735e88cea0767f4e9e3788b8027d80b306103750d1f0de89fc7"
    params = {
        "n_shots": 0,
        "random_seed": 0,
        "output_pattern": "{{target}}",
        "system_message": None,
        "metadata_keys": "target",
        "prompt_pattern": "[{\"role\": \"user\", \"content\": \"{{input}}\"}]",
        "few_shot_pattern": None,
        "few_shot_separator": None,
        "prefix": "[{\"role\": \"user\", \"content\": \"Answer questions about causal attribution.\"}, {\"role\": \"user\", \"content\": \"How would a typical person answer each of the following questions about causation?\\nFrank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbors body, causing significant injury. Did Frank T. intentionally shoot his neighbor in the body?\\nOptions:\\n- Yes\\n- No\"}, {\"role\": \"assistant\", \"content\": \"Lets think step by step.\\nHere in this question, we are told that \\\"Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild.\\\" A typical person would assume that this passage suggests that Frank T. had no intention of shooting and injuring someone and that the bullet accidentally hit the neighbors body; therefore, we conclude that Frank T. did not intentionally hit his neighbor. So the answer is No.\"}, {\"role\": \"user\", \"content\": \"How would a typical person answer each of the following questions about causation?\\nSuzy and Billy are working on a project that is very important for our nations security. The boss tells them both: \\\"Be sure that you are here at exactly 9 am. It is absolutely essential that you arrive at that time.\\\" Both Billy and Suzy arrive at 9 am. As it happens, there was a motion detector installed in the room where they arrived. The motion detector was set up to be triggered if at least one person appeared in the room at the same time. So the motion detector went off. Did Billy cause the motion detector to go off?\\nOptions:\\n- Yes\\n- No\"}, {\"role\": \"assistant\", \"content\": \"Lets think step by step.\\nHere in this question, we are told that the boss ordered them both to arrive at the meeting room at the same time and that the motion detector was set up to be triggered if at least one person appeared in the room at the same time.\\\" A typical person would assume that the person probably meant to say the detector was set up to be triggered if \\\"both persons\\\" appeared in the room at the same time, not at least one person, since otherwise the phrase \\\"at the same time\\\" would not make much sense in that sentence. Because the motion detector went off, a typical person would therefore come to the conclusion that both Suzy and Billy triggered the motion detector to go off; hence, Billy did indeed cause the motion detector to go off. So the answer is Yes.\"}, {\"role\": \"user\", \"content\": \"How would a typical person answer each of the following questions about causation?\\nGeorge and his sister Lena reunite at their parents house for Thanksgiving. Whereas George just got into medical school, Lena is unhappy in her marriage and recently lost her job. Over the course of the day, George and Lena get into a number of heated arguments. Later in the afternoon they play a game of darts. They split the first two games, and the third game is close until the end. Who will win comes down to Georges last shot. If he hits a high point region, he wins; if he hits a low point region, Lena wins. George thinks of the difficult time Lena is having, and he really wants to let her win. He aims the dart at the low point region. He sets up his shot and the dart lands in the low point region. After his shot, Lena wins the game and is very happy. Did George hit the low point region intentionally?\\nOptions:\\n- Yes\\n- No\"}, {\"role\": \"assistant\", \"content\": \"Lets think step by step.\\nHere in this question, we are told that \\\"He aims the dart at the low point region.\\\" A typical person might therefore think George did intentionally hit the low point region, because he wanted to lift up the spirit of his sister Lena. So the answer is Yes.\"}]",  # noqa: E501
        "label_map": None,
        "additional_payload": None,
    }
    base_test("bigbench-hard_causal_judgement", completion_ground_truth_checksum, chat_ground_truth_checksum, params)

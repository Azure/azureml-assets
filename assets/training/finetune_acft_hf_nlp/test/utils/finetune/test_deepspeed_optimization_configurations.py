# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing test cases for invalid user settings for deepspeed stage3."""

import json
import pytest
from unittest import TestCase
from argparse import Namespace
from typing import Dict, Any

from src.finetune.finetune import (
    validate_ds_zero3_config,
    check_for_invalid_ds_zero3_settings,
    identify_deepspeed_stage,
    setup_and_validate_deepspeed,
    resolve_deepspeed_config,
    DEFAULT_DEEPSPEED_STAGE2_CONFIG,
    DEFAULT_DEEPSPEED_STAGE3_CONFIG,
)

from azureml.acft.contrib.hf.nlp.constants.constants import Tasks, HfModelTypes


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read json file."""
    with open(file_path, 'r', encoding='utf-8') as rptr:
        file_path_dict = json.load(rptr)
    return file_path_dict


def test_identify_deepspeed_stage():
    """Test if the right deepspeed stage is getting identified."""
    assert identify_deepspeed_stage(read_json_file("test/utils/finetune/data/valid_ds3_config.json")) == 3


def test_validate_ds_zero3_config():
    """Validate deepspeed config parameters passed in by the user."""
    # catch the error in the invalid case
    ut_obj = TestCase()
    with ut_obj.assertRaises(Exception) as context:
        validate_ds_zero3_config(read_json_file("test/utils/finetune/data/invalid_ds3_config.json"))

    ut_obj.assertTrue(
        (
            "stage3_gather_16bit_weights_on_model_save should be `true` "
            "in deepspeed stage 3 config."
        )
        in str(context.exception)
    )

    # valid case
    validate_ds_zero3_config(read_json_file("test/utils/finetune/data/valid_ds3_config.json"))


def test_invalid_fail_run_setting_ds3_ort():
    """Test can_apply_ort return true for generation task."""
    args = Namespace(apply_ort=True, apply_deepspeed=True, deepspeed="test/utils/finetune/data/valid_ds3_config.json")
    ut_obj = TestCase()
    with ut_obj.assertRaises(Exception) as context:
        check_for_invalid_ds_zero3_settings(args)

    ut_obj.assertTrue(
        f"Invalid settings found. Deep Speed stage3 doesn't work with {dict(apply_ort=True)}"
        in str(context.exception)
    )


def test_invalid_fail_run_setting_ds3_lora():
    """Test can_apply_ort return true for generation task."""
    args = Namespace(
        apply_lora=True,
        apply_deepspeed=True,
        deepspeed="test/utils/finetune/data/valid_ds3_config.json",
        task_name=Tasks.SINGLE_LABEL_CLASSIFICATION,
        model_type=HfModelTypes.GPT2,
    )
    ut_obj = TestCase()
    with ut_obj.assertRaises(Exception) as context:
        check_for_invalid_ds_zero3_settings(args)

    ut_obj.assertTrue(
        "Invalid settings found. Deep Speed stage3 doesn't work with "
        in str(context.exception)
    )


def test_valid_run_setting_ds3_lora():
    """Test can_apply_ort return true for generation task."""
    args = Namespace(
        apply_lora=True,
        apply_deepspeed=True,
        deepspeed="test/utils/finetune/data/valid_ds3_config.json",
        task_name=Tasks.TEXT_GENERATION,
        model_type=HfModelTypes.LLAMA,
    )
    try:
        check_for_invalid_ds_zero3_settings(args)
    except Exception as e:
        pytest.fail(str(e))


def test_invalid_setting_ds3_auto_find_bs():
    """Test can_apply_ort return true for generation task."""
    args = Namespace(
        auto_find_batch_size=True,
        apply_deepspeed=True,
        deepspeed="test/utils/finetune/data/valid_ds3_config.json"
    )
    check_for_invalid_ds_zero3_settings(args)

    assert getattr(args, "auto_find_batch_size") is False


def test_enable_gradient_checkpointing():
    """Test can_apply_ort return true for generation task."""
    args = Namespace(
        model_type="llama",
        apply_deepspeed=True,
        deepspeed="test/utils/finetune/data/valid_ds3_config.json"
    )
    setup_and_validate_deepspeed(args, do_validate=True)

    assert getattr(args, "gradient_checkpointing") is True


def test_default_deepspeed_config():
    """Test if the default deepspeed config is applied or not."""
    # Check for default stage2 config
    args = Namespace(apply_deepspeed=True, deepspeed_stage=2, deepspeed=None)
    assert resolve_deepspeed_config(args) is DEFAULT_DEEPSPEED_STAGE2_CONFIG

    # Check for default stage3 config
    args = Namespace(apply_deepspeed=True, deepspeed_stage=3, deepspeed=None)
    assert resolve_deepspeed_config(args) is DEFAULT_DEEPSPEED_STAGE3_CONFIG

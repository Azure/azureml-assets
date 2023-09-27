# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing test cases for invalid user settings for deepspeed stage3."""

from unittest import TestCase
from src.finetune.finetune import (
    validate_ds_zero3_config,
    check_for_invalid_ds_zero3_settings,
    identify_deepspeed_stage
)
from argparse import Namespace


def test_identify_deepspeed_stage():
    """Test if the right deepspeed stage is getting identified."""
    assert identify_deepspeed_stage("test/utils/finetune/data/valid_ds3_config.json") == 3


def test_validate_ds_zero3_config():
    """Validate deepspeed config parameters passed in by the user."""

    # catch the error in the invalid case
    ut_obj = TestCase()
    with ut_obj.assertRaises(Exception) as context:
        validate_ds_zero3_config("test/utils/finetune/data/invalid_ds3_config.json")

    ut_obj.assertTrue(
        (
            "stage3_gather_16bit_weights_on_model_save should be `true` "
            "in deepspeed stage 3 config."
        )
        in str(context.exception)
    )

    # valid case
    validate_ds_zero3_config("test/utils/finetune/data/valid_ds3_config.json")


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
    args = Namespace(apply_lora=True, apply_deepspeed=True, deepspeed="test/utils/finetune/data/valid_ds3_config.json")
    ut_obj = TestCase()
    with ut_obj.assertRaises(Exception) as context:
        check_for_invalid_ds_zero3_settings(args)

    ut_obj.assertTrue(
        f"Invalid settings found. Deep Speed stage3 doesn't work with {dict(apply_lora=True)}"
        in str(context.exception)
    )


def test_invalid_setting_ds3_auto_find_bs():
    """Test can_apply_ort return true for generation task."""
    args = Namespace(auto_find_batch_size=True, apply_deepspeed=True, deepspeed="./data/valid_ds3_config.json")
    check_for_invalid_ds_zero3_settings(args)

    assert getattr(args, "auto_find_batch_size") is False

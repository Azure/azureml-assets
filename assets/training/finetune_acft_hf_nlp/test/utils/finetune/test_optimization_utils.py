# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing test cases for utils function from optimization_utils."""

from azureml.acft.contrib.hf.nlp.constants.constants import Tasks
from src.utils.finetune.optimization_utils import can_apply_ort
from argparse import Namespace
import logging


def test_can_apply_ort_return_false_summarization_task():
    """Test can_apply_ort return false for summarization task."""
    args = Namespace(apply_ort=True, task_name=Tasks.SUMMARIZATION)
    logger = logging.getLogger(__name__)
    assert can_apply_ort(args, logger) is False


def test_can_apply_ort_return_false_translation_task():
    """Test can_apply_ort return false for translation task."""
    args = Namespace(apply_ort=True, task_name=Tasks.TRANSLATION)
    logger = logging.getLogger(__name__)
    assert can_apply_ort(args, logger) is False


def test_can_apply_ort_return_true_text_generation_task():
    """Test can_apply_ort return true for generation task."""
    args = Namespace(apply_ort=True, task_name=Tasks.TEXT_GENERATION)
    logger = logging.getLogger(__name__)
    assert can_apply_ort(args, logger) is True


def test_can_apply_ort_return_true_text_classification_task():
    """Test can_apply_ort return true for text_classification task."""
    args = Namespace(apply_ort=True, task_name=Tasks.SINGLE_LABEL_CLASSIFICATION)
    logger = logging.getLogger(__name__)
    assert can_apply_ort(args, logger) is True


def test_can_apply_ort_return_true_qanda_task():
    """Test can_apply_ort return true for qanda task."""
    args = Namespace(apply_ort=True, task_name=Tasks.QUESTION_ANSWERING)
    logger = logging.getLogger(__name__)
    assert can_apply_ort(args, logger) is True

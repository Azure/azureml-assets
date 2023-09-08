from azureml.acft.contrib.hf.nlp.constants.constants import Tasks
from src.utils.finetune.optimization_utils import should_apply_ort
from argparse import Namespace
import logging


def test_should_apply_ort_return_false_summarization_task():
    args = Namespace(apply_ort=True, task_name=Tasks.SUMMARIZATION)
    logger = logging.getLogger(__name__)
    assert should_apply_ort(args, logger) == False


def test_should_apply_ort_return_false_translation_task():
    args = Namespace(apply_ort=True, task_name=Tasks.TRANSLATION)
    logger = logging.getLogger(__name__)
    assert should_apply_ort(args, logger) == False


def test_should_apply_ort_return_true_text_generation_task():
    args = Namespace(apply_ort=True, task_name=Tasks.TEXT_GENERATION)
    logger = logging.getLogger(__name__)
    assert should_apply_ort(args, logger) == True


def test_should_apply_ort_return_true_text_classification_task():
    args = Namespace(apply_ort=True, task_name=Tasks.SINGLE_LABEL_CLASSIFICATION)
    logger = logging.getLogger(__name__)
    assert should_apply_ort(args, logger) == True


def test_should_apply_ort_return_true_qanda_task():
    args = Namespace(apply_ort=True, task_name=Tasks.QUESTION_ANSWERING)
    logger = logging.getLogger(__name__)
    assert should_apply_ort(args, logger) == True
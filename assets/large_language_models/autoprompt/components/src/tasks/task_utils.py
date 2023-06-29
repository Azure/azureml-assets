"""Task Utilities."""

from tasks.abstractive import AbstractiveQA
from tasks.mcq import MCQ
from tasks.arithmetic import ArithmeticQA
from constants import TaskTypes


def get_task(qa_task, meta_prompts=[], num_choices=5, n_prompts=10, best_of=100, primary_metric="f1_score"):
    """Get Task."""
    if qa_task == TaskTypes.arithmetic:
        return ArithmeticQA(meta_prompts)

    if qa_task == TaskTypes.mcq:
        return MCQ(meta_prompts, num_choices)

    if qa_task == TaskTypes.abstractive:
        return AbstractiveQA(meta_prompts, n_prompts, best_of, primary_metric)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Global test fixtures for evaluator tests."""

import sys
from pathlib import Path


def get_evaluator_path(evaluator_name: str) -> Path:
    """Get the path to an evaluator's code directory.

    :param evaluator_name: Name of the evaluator (e.g., 'regex_match')
    :type evaluator_name: str
    :return: Path to the evaluator directory
    :rtype: Path
    """
    return Path(__file__).parent.parent / evaluator_name / "evaluator"


def _setup_evaluator_paths():
    """Add evaluator paths to sys.path for importing evaluator modules."""
    evaluators_with_tests = ["regex_match"]

    for evaluator_name in evaluators_with_tests:
        evaluator_path = get_evaluator_path(evaluator_name)
        if evaluator_path.exists() and str(evaluator_path) not in sys.path:
            sys.path.insert(0, str(evaluator_path))


_setup_evaluator_paths()

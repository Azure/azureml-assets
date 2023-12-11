# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing constants for model evaluation script."""


class TASK:
    """TASK list."""

    CLASSIFICATION = "tabular-classification"
    CLASSIFICATION_MULTILABEL = "tabular-classification-multilabel"
    REGRESSION = "tabular-regression"


ALL_TASKS = [
    TASK.CLASSIFICATION,
    TASK.CLASSIFICATION_MULTILABEL,
    TASK.REGRESSION
]

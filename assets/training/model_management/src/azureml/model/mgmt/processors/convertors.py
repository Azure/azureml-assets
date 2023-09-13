# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow model convertor interface."""

from abc import ABC, abstractmethod


class MLFLowConvertorInterface(ABC):
    """MLflow convertor interface."""

    @abstractmethod
    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        raise NotImplementedError

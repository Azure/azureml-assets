# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory module to create task based convertor classes."""

from abc import ABC, abstractmethod
from azureml.model.mgmt.processors.pyfunc.config import (
    SupportedVisionTasks,
)
from azureml.model.mgmt.utils.logging_utils import get_logger
from .convertors import (
    VisionMLFlowConvertor,
)


logger = get_logger(__name__)


def get_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
    """Instantiate and return PyFunc MLflow convertor."""
    task = translate_params["task"]
    if SupportedVisionTasks.has_value(task):
        return VisionMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
    else:
        raise Exception(f"{task} not supported for MLflow conversion using pyfunc flavor.")


class PyFuncMLflowConvertorFactoryInterface(ABC):
    """PyFunc MLflow covertor factory interface."""

    @abstractmethod
    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor."""
        raise NotImplementedError


class VisionMLflowConvertorFactory(PyFuncMLflowConvertorFactoryInterface):
    """Factory class for vision model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for vision tasks."""
        return VisionMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The base configuration class for all type of models."""


from abc import ABC, abstractmethod
from typing import List, Dict

from constants import EngineName


class ModelConfigurationBuilder(ABC):
    """Base class for model configurations builder for optimization engine configuration."""

    MLFLOW_MODEL_PATH = "mlflow_model_folder/data/model"

    @property
    def model_path(self) -> str:
        """Returns the relative path of the mlflow model.

        :return: Relative path of the mlflow model.
        :rtype: str
        """
        return self.MLFLOW_MODEL_PATH

    @property
    def engine(self) -> EngineName:
        """Inference optimization engine.

        :return: Optimization engine to use.
        :rtype: EngineName enum
        """
        return EngineName.VLLM

    @abstractmethod
    def post_processing(self, responses: any, inference_time: float, **kwargs) -> List[any]:
        """Process model output to generate the final response.

        :param responses: model responses
        :type responses: any
        :param inference_time: inference time in milliseconds
        :type inference_time: float
        :return: final response
        :rtype: List of any
        """
        pass

    @abstractmethod
    def get_task_config(self) -> Dict[str, str]:
        """Get task configuration.

        :return: task configuration
        :rtype: Dict[str, str]
        """
        pass

    @abstractmethod
    def get_optimization_config(self) -> Dict[str, str]:
        """Get optimization configuration.

        :return: optimization configuration
        :rtype: Dict[str, str]
        """
        pass

    @abstractmethod
    def update_mii_model_config(self, model_config: object) -> None:
        """Override the default MII model configuration.

        :param model_config: MII model configuration
        :type model_config: mii.ModelConfig
        """
        pass

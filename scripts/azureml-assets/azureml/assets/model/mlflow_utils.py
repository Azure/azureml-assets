# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MLFlow Operations Class."""


class MLFlowModelUtils:
    """Transform Model to MLFlow Model."""

    MLMODEL_FILE_NAME = "MLmodel"
    MLFLOW_MODEL_PATH = "mlflow_model_folder"

    def __init__(self, name, task_name, flavor, mlflow_model_dir):
        """Initialize object for MLFlowModelUtils."""
        self.name = name
        self.task_name = task_name
        self.mlflow_model_dir = mlflow_model_dir
        self.flavor = flavor

    def _convert_to_mlflow_hftransformers(self):
        """Convert the model using MLFlow Huggingface Flavor."""
        # TODO Add MLFlow HFFlavour support once wheel files are publicly available
        return False

    def _convert_to_mlflow_package(self):
        """Convert the model using pyfunc flavor."""
        return False

    def convert_into_mlflow_model(self):
        """Convert the model with given flavor."""
        if self.flavor == "hftransformers":
            return self._convert_to_mlflow_hftransformers()
        # TODO add support for pyfunc. Pyfunc requires custom env file.
        else:
            return self._convert_to_mlflow_package()

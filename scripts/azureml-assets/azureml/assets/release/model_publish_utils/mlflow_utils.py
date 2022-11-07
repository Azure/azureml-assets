# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class MLFlowModelUtils:
    def __init__(self, name, task_name, flavor, mlflow_model_dir):
        self.name = name
        self.task_name = task_name
        self.mlflow_model_dir = mlflow_model_dir
        self.flavor = flavor

    def _convert_to_mlflow_hftransformers(self):
        # TODO Add MLFlow HFFlavour support once wheel files are publicly available
        return None

    def _convert_to_mlflow_package(self):
        return None

    def covert_into_mlflow_model(self):
        if self.flavor == "hftransformers":
            self._convert_to_mlflow_hftransformers()
        # TODO add support for pyfunc. Pyfunc requires custom env file.
        else:
            self._convert_to_mlflow_package()

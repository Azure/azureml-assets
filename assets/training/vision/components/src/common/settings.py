# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Classes for ML tasks."""


import argparse
from azureml.automl.core.shared.constants import Tasks
from azureml.automl.dnn.vision.common import utils


class CommonSettings:
    """CommonSettings class."""

    def __init__(self, training_data: str, validation_data: str, mlflow_model_output: str,
                 pytorch_model_output: str) -> None:
        """Init function for CommonSettings class."""
        self.training_data = training_data
        self.validation_data = validation_data
        self.mlflow_model_output = mlflow_model_output
        self.pytorch_model_output = pytorch_model_output

    @classmethod
    def create_from_parsing_current_cmd_line_args(cls) -> "CommonSettings":
        """Create object from parsing current cmd line args."""
        parser = argparse.ArgumentParser()
        parser.add_argument(utils._make_arg('training_data'), type=str)
        parser.add_argument(utils._make_arg('validation_data'), type=str)
        parser.add_argument(utils._make_arg('mlflow_model_output'), type=str)
        parser.add_argument(utils._make_arg('pytorch_model_output'), type=str)
        args, _ = parser.parse_known_args()
        return CommonSettings(
            args.training_data, args.validation_data, args.mlflow_model_output, args.pytorch_model_output
        )


class ClassificationSettings(CommonSettings):
    """ClassificationSettings class."""

    def __init__(self, training_data: str, validation_data: str, mlflow_model_output: str,
                 pytorch_model_output: str, task_type: str) -> None:
        """Init function for ClassificationSettings class."""
        super().__init__(training_data, validation_data, mlflow_model_output, pytorch_model_output)
        self.multilabel = False
        if task_type == Tasks.IMAGE_CLASSIFICATION_MULTILABEL:
            self.multilabel = True

    @classmethod
    def create_from_parsing_current_cmd_line_args(cls) -> "ClassificationSettings":
        """Create object from parsing current cmd line args."""
        # Create common settings
        common_settings = CommonSettings.create_from_parsing_current_cmd_line_args()

        # Create classification settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--task_type', type=str)
        args, _ = parser.parse_known_args()
        return ClassificationSettings(
            common_settings.training_data, common_settings.validation_data, common_settings.mlflow_model_output,
            common_settings.pytorch_model_output, args.task_type)


class ObjectDetectionSettings(CommonSettings):
    """ObjectDetectionSettings class."""

    pass


class InstanceSegmentationSettings(CommonSettings):
    """InstanceSegmentationSettings class."""

    pass

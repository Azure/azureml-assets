# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from azureml.automl.core.shared.constants import Tasks
from azureml.automl.dnn.vision.common import utils


class CommonSettings:

    def __init__(self, training_data: str, validation_data: str, mlflow_model_output: str, pytorch_model_output: str) -> None:
        self.training_data = training_data
        self.validation_data = validation_data
        self.mlflow_model_output = mlflow_model_output
        self.pytorch_model_output = pytorch_model_output

    @classmethod
    def create_from_parsing_current_cmd_line_args(cls) -> "CommonSettings":
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

    def __init__(self, training_data: str, validation_data: str, mlflow_model_output: str, pytorch_model_output: str, task_type: str) -> None:
        super().__init__(training_data, validation_data, mlflow_model_output, pytorch_model_output)
        self.multilabel = False
        if task_type == Tasks.IMAGE_CLASSIFICATION_MULTILABEL:
            self.multilabel = True

    @classmethod
    def create_from_parsing_current_cmd_line_args(cls) -> "ClassificationSettings":
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
    pass


class InstanceSegmentationSettings(CommonSettings):
    pass

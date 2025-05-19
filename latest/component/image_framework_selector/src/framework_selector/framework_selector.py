# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
File containing function for model framework selector component for images.

Return true for runtime image command component and False for using FT component.

TODO: Framework selectors to use "switch" control flow once its supported by pipeline team.
"""

from mldesigner import Input, Output, command_component
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.common_components import (
    get_logger_app,
    set_logging_parameters,
    LoggingLiterals,
    PROJECT_NAME,
    VERSION
)
from azureml.acft.common_components.model_selector.constants import ModelRepositoryURLs
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException

logger = get_logger_app("azureml.acft.common_components.scripts.components.framework_selector.framework_selector")


class Tasks:
    """Define types of machine learning tasks supported by automated ML."""

    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_OBJECT_DETECTION = "image-object-detection"
    IMAGE_INSTANCE_SEGMENTATION = "image-instance-segmentation"


class RuntimeModels:
    """Define types of machine learning tasks supported by automated ML."""

    IMAGE_CLASSIFICATION = [
        "mobilenetv2",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnest50",
        "resnest101",
        "seresnext",
        "vits16r224",
        "vitb16r224",
        "vitl16r224"
    ]

    IMAGE_OBJECT_DETECTION = [
        "yolov5",
        "fasterrcnn_resnet18_fpn",
        "fasterrcnn_resnet34_fpn",
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet101_fpn",
        "fasterrcnn_resnet152_fpn",
        "retinanet_resnet50_fpn"
    ]

    IMAGE_INSTANCE_SEGMENTATION = [
        "maskrcnn_resnet18_fpn",
        "maskrcnn_resnet34_fpn",
        "maskrcnn_resnet50_fpn",
        "maskrcnn_resnet101_fpn",
        "maskrcnn_resnet152_fpn"
    ]

    def get_supported_task_type_for_model(self, model_name: str) -> str:
        """Return the supported task type for the given model.

        :param model_name: Name of the model.
        :type model_name: str

        :return: Supported automl task type for the given model.
        :rtype: str
        """
        if model_name in self.IMAGE_CLASSIFICATION:
            return Tasks.IMAGE_CLASSIFICATION

        if model_name in self.IMAGE_OBJECT_DETECTION:
            return Tasks.IMAGE_OBJECT_DETECTION

        if model_name in self.IMAGE_INSTANCE_SEGMENTATION:
            return Tasks.IMAGE_INSTANCE_SEGMENTATION

        logger.info(f"Model {model_name} is not supported by automl image runtime component.")
        return None

    def get_supported_models_for_task_type(self, task_type: str) -> list:
        """Return the supported models for the given task type.

        :param task_type: Type of the task.
        :type task_type: str

        :return: Supported models for the given task type.
        :rtype: list
        """
        if task_type == Tasks.IMAGE_CLASSIFICATION:
            return self.IMAGE_CLASSIFICATION

        if task_type == Tasks.IMAGE_OBJECT_DETECTION:
            return self.IMAGE_OBJECT_DETECTION

        if task_type == Tasks.IMAGE_INSTANCE_SEGMENTATION:
            return self.IMAGE_INSTANCE_SEGMENTATION

        logger.info(f"Task type {task_type} is not supported by automl image runtime component.")
        return None

    def check_model_support_for_runtime_task(self, model_name: str, task_type: str) -> bool:
        """Return true if model is supported by runtime image command component.

        :param model_name: Name of the model.
        :type model_name: str
        :param task_type: Type of the runtime task.
        :type task_type: str

        :return: True if model is supported by runtime image command component.
        :rtype: bool
        """
        if task_type is None:
            return False
        model_supported_task_type = self.get_supported_task_type_for_model(model_name)
        if model_supported_task_type is not None and task_type.lower() == model_supported_task_type.lower():
            logger.info(f"{model_name} is in the list of supported models by runtime. "
                        f"Using runtime {task_type} command component.")
            return True
        elif model_supported_task_type is not None:
            supported_models = self.get_supported_models_for_task_type(task_type)
            error_string = f"The selected automl model({model_name}) doesn't support {task_type}. " \
                f"{model_name} only supports {model_supported_task_type}. " \
                f"Task type selected = {task_type}. " \
                f"Automl Models that support {task_type} are [{', '.join(supported_models)}]."

            if task_type == Tasks.IMAGE_CLASSIFICATION:
                error_string += f" Additionally, you can also select a model from hugging face. Please check " \
                                f"{ModelRepositoryURLs.HF_TRANSFORMER_IMAGE_CLASSIFIFCATION} for " \
                                f"valid model name."
            else:
                error_string += f" Additionally, you can also select a model from mmdetection model zoo. " \
                                f"To find the correct model name, go to {ModelRepositoryURLs.MMDETECTION}, " \
                                f"click on the model type, and you will find the " \
                                f"model name in the metafile.yml file which is present at " \
                                "configs/<MODEL_TYPE>/metafile.yml location."

            raise ACFTValidationException._with_error(
                AzureMLError.create(ACFTUserError, pii_safe_message=error_string))

        acft_component_name = task_type.replace("-", " ")
        logger.info(f"Model {model_name} is not supported by automl image runtime component. \
                    Using finetune {acft_component_name} component.")
        return False


@command_component()
def framework_selector(task_type: str, model_name: Input(type="string", optional=True)) \
        -> Output(type="boolean", is_control=True):  # noqa: F821
    """Return true if model is supported by runtime image command component.

    :param task_type: Type of the task.
    :type task_type: str
    :param model_name: Name of the model.
    :type model_name: str

    :return: True if model is supported by runtime image command component.
    :rtype: bool
    """
    set_logging_parameters(
        task_type="framework_selector-" + task_type,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
        },
    )

    # By default run models from runtime.
    if not model_name:
        logger.info("model_name is None. Using default models from runtime image command component.")
        return True

    if task_type in [Tasks.IMAGE_CLASSIFICATION, Tasks.IMAGE_OBJECT_DETECTION, Tasks.IMAGE_INSTANCE_SEGMENTATION]:
        return RuntimeModels().check_model_support_for_runtime_task(model_name, task_type)
    else:
        return True

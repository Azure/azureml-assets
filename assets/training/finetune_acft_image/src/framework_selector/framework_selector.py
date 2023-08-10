# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
File containing function for model framework selector component for images.

Return true for runtime image command component and False for using FT component.

TODO: Framework selectors to use "switch" control flow once its supported by pipeline team.
"""

from azure.ai.ml import Input
from mldesigner import Output, command_component
from azureml.acft.common_components import get_logger_app, set_logging_parameters, \
    LoggingLiterals, PROJECT_NAME, VERSION

logger = get_logger_app("azureml.acft.common_components.scripts.components.framework_selector.framework_selector")


class Tasks:
    """Define types of machine learning tasks supported by automated ML."""

    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_OBJECT_DETECTION = "image-object-detection"
    IMAGE_INSTANCE_SEGMENTATION = "image-instance-segmentation"


def image_classification_framework_selector(model_name: str) -> bool:
    """Return true if model is supported by runtime image command component.

    :param model_name: Name of the model.
    :type model_name: str

    :return: True if model is supported by runtime image command component.
    :rtype: bool
    """
    # TODO: Temporary list for now.
    image_classification_models_runtime = [
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

    if (model_name in image_classification_models_runtime):
        logger.info(f"{model_name} is in the list of supported models by runtime. "
                    "Using runtime image classification command component.")
        return True

    logger.info(f"{model_name} is not in the list of supported models by runtime. "
                "Using finetune image classification component.")
    return False


def image_object_detection_framework_selector(model_name: str):
    """Return true if model is supported by runtime image command component.

    :param model_name: Name of the model.
    :type model_name: str

    :return: True if model is supported by runtime image command component.
    :rtype: bool
    """
    # TODO: Temporary list for now.
    image_object_detection_models_runtime = [
        "yolov5",
        "fasterrcnn_resnet18_fpn",
        "fasterrcnn_resnet34_fpn",
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet101_fpn",
        "fasterrcnn_resnet152_fpn",
        "retinanet_resnet50_fpn"
    ]

    if (model_name in image_object_detection_models_runtime):
        logger.info(f"{model_name} is in the list of supported models by runtime. "
                    "Using runtime image object detection command component.")
        return True

    logger.info(f"{model_name} is not in the list of supported models by runtime. "
                "Using finetune image object detection component.")
    return False


def image_instance_segmentation_framework_selector(model_name: str):
    """Return true if model is supported by runtime image command component.

    :param model_name: Name of the model.
    :type model_name: str

    :return: True if model is supported by runtime image command component.
    :rtype: bool
    """
    # TODO: Temporary list for now.
    image_instance_segmentation_models_runtime = [
        "maskrcnn_resnet18_fpn",
        "maskrcnn_resnet34_fpn",
        "maskrcnn_resnet50_fpn",
        "maskrcnn_resnet101_fpn",
        "maskrcnn_resnet152_fpn"
    ]

    if (model_name in image_instance_segmentation_models_runtime):
        logger.info(f"{model_name} is in the list of supported models by runtime. "
                    "Using runtime image instance segmentation command component.")
        return True

    logger.info(f"{model_name} is not in the list of supported models by runtime. "
                "Using finetune image instance segmentation component.")
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

    if task_type == Tasks.IMAGE_CLASSIFICATION:
        return image_classification_framework_selector(model_name)

    elif task_type == Tasks.IMAGE_OBJECT_DETECTION:
        return image_object_detection_framework_selector(model_name)

    elif task_type == Tasks.IMAGE_INSTANCE_SEGMENTATION:
        return image_instance_segmentation_framework_selector(model_name)

    else:
        return True

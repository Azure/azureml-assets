# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory module to create task based convertor classes."""

from abc import ABC, abstractmethod
from azureml.model.mgmt.config import ModelFramework
from azureml.model.mgmt.processors.transformers.config import (
    SupportedASRModelFamily,
    SupportedDiffusersTask,
    SupportedNLPTasks,
    SupportedTasks,
    SupportedTextToImageModelFamily,
    SupportedVisionTasks,
)
from azureml.model.mgmt.processors.pyfunc.config import (
    MMLabDetectionTasks,
    SupportedTasks as PyFuncSupportedTasks
)
from azureml.model.mgmt.utils.logging_utils import get_logger
from azureml.model.mgmt.processors.transformers.convertors import (
    NLPMLflowConvertor,
    VisionMLflowConvertor,
    WhisperMLflowConvertor,
)
from azureml.model.mgmt.processors.pyfunc.convertors import (
    MMLabDetectionMLflowConvertor,
    CLIPMLFlowConvertor,
    StableDiffusionMlflowConvertor,
    LLaVAMLFlowConvertor,
)


logger = get_logger(__name__)


def get_mlflow_convertor(model_framework, model_dir, output_dir, temp_dir, translate_params):
    """Instantiate and return MLflow convertor."""
    task = translate_params["task"]

    if model_framework == ModelFramework.HUGGINGFACE.value:
        # Models from Hugging face framework exported in transformers mlflow flavor
        if SupportedNLPTasks.has_value(task):
            return NLPMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        elif SupportedVisionTasks.has_value(task):
            return VisionMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif SupportedDiffusersTask.has_value(task):
            return DiffusersMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif task == SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value:
            return ASRMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        # Models from Hugging face framework exported in PyFunc mlflow flavor
        elif task == PyFuncSupportedTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value:
            return CLIPMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        else:
            raise Exception(f"Models from {model_framework} for {task} not supported for MLflow conversion")
    elif model_framework == ModelFramework.MMLAB.value:
        # Models from MMLAB model framework exported in PyFunc mlflow flavor
        if MMLabDetectionTasks.has_value(task):
            return MMLabDetectionMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        else:
            raise Exception(f"Models from {model_framework} for {task} not supported for MLflow conversion")
    elif model_framework == ModelFramework.LLAVA.value:
        # Models from LLAVA model framework exported in PyFunc mlflow flavor
        if task == PyFuncSupportedTasks.IMAGE_TEXT_TO_TEXT.value:
            return LLaVAMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        else:
            raise Exception(f"Models from {model_framework} for {task} not supported for MLflow conversion")
    else:
        raise Exception(f"Models from {model_framework} not supported for MLflow conversion")


class MLflowConvertorFactoryInterface(ABC):
    """MLflow covertor factory interface."""

    @abstractmethod
    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor."""
        raise NotImplementedError


class NLPMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for NLP model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for NLP tasks."""
        return NLPMLflowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class VisionMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for vision model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for vision tasks."""
        return VisionMLflowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class ASRMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for ASR model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for ASR tasks."""
        misc = translate_params["misc"]
        if misc and SupportedASRModelFamily.WHISPER.value in misc:
            return WhisperMLflowConvertor(
                model_dir=model_dir,
                output_dir=output_dir,
                temp_dir=temp_dir,
                translate_params=translate_params,
            )
        raise Exception("Unsupported ASR model family")


class DiffusersMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for diffusor model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for diffusers."""
        misc = translate_params["misc"]
        if misc and SupportedTextToImageModelFamily.STABLE_DIFFUSION.value in misc:
            return StableDiffusionMlflowConvertor(
                model_dir=model_dir,
                output_dir=output_dir,
                temp_dir=temp_dir,
                translate_params=translate_params,
            )
        raise Exception("Unsupported diffuser model family")


class MMLabDetectionMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for MMLab detection model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for vision tasks."""
        return MMLabDetectionMLflowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class CLIPMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for clip model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for CLIP model."""
        return CLIPMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class LLaVAMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for LLaVA model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for LLaVA model."""
        return LLaVAMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory module to create task based convertor classes."""

from abc import ABC, abstractmethod
from azureml.model.mgmt.processors.transformers.config import (
    SupportedASRModelFamily,
    SupportedDiffusersTask,
    SupportedNLPTasks,
    SupportedTasks,
    SupportedTextToImageModelFamily,
    SupportedVisionTasks,
)
from .convertors import (
    NLPMLflowConvertor,
    VisionMLflowConvertor,
    WhisperMLflowConvertor,
    StableDiffusionMlflowConvertor,
)


def get_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
    """Instantiate and return hftransformers MLflow convertor."""
    task = translate_params["task"]
    if SupportedNLPTasks.has_value(task):
        return NLPMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
    elif SupportedVisionTasks.has_value(task):
        return VisionMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
    elif SupportedDiffusersTask.has_value(task):
        return DiffusersMLflowConvertorFactory.create_mlflow_convertor(
            model_dir, output_dir, temp_dir, translate_params
        )
    elif task == SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value:
        return ASRMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
    else:
        raise Exception(f"{task} not supported for MLflow conversion using hftransformers")


class HFMLflowConvertorFactoryInterface(ABC):
    """HF MLflow covertor factory interface."""

    @abstractmethod
    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor."""
        raise NotImplementedError


class NLPMLflowConvertorFactory(HFMLflowConvertorFactoryInterface):
    """Factory class for NLP model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for NLP tasks."""
        return NLPMLflowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class VisionMLflowConvertorFactory(HFMLflowConvertorFactoryInterface):
    """Factory class for vision model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for vision tasks."""
        return VisionMLflowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class ASRMLflowConvertorFactory(HFMLflowConvertorFactoryInterface):
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


class DiffusersMLflowConvertorFactory(HFMLflowConvertorFactoryInterface):
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

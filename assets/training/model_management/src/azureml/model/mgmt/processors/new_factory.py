# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory module to create task based convertor classes."""

from abc import ABC, abstractmethod
from azureml.model.mgmt.config import ModelFramework
from azureml.model.mgmt.processors.transformers.config import (
    SupportedASRModelFamily,
    SupportedNLPTasks,
    SupportedTasks,
    SupportedVisionTasks,
)
from azureml.model.mgmt.processors.pyfunc.config import (
    MMLabDetectionTasks,
    MMLabTrackingTasks,
    ModelFamilyPrefixes,
    SupportedTasks as PyFuncSupportedTasks,
)
from azureml.model.mgmt.processors.pyfunc.text_to_image.config import SupportedTextToImageModelFamily
from azureml.model.mgmt.utils.logging_utils import get_logger
from azureml.model.mgmt.processors.transformers.convertors import (
    NLPMLflowConvertor,
    VisionMLflowConvertor,
    WhisperMLflowConvertor,
)
from azureml.model.mgmt.processors.pyfunc.convertors import (
    AutoMLMLFlowConvertor,
    BLIPMLFlowConvertor,
    MMLabDetectionMLflowConvertor,
    MMLabTrackingMLflowConvertor,
    CLIPMLFlowConvertor,
    StableDiffusionMlflowConvertor,
    StableDiffusionInpaintingMlflowConvertor,
    StableDiffusionImageToImageMlflowConvertor,
    DinoV2MLFlowConvertor,
    LLaVAMLFlowConvertor,
    SegmentAnythingMLFlowConvertor,
    VirchowMLFlowConvertor
)


logger = get_logger(__name__)

MLFLOW_CONVERTER_REGISTRY = {}

def register_mlflow_converter(model_framework, canonical_task, model_family=None):
    """
    Decorator to register an MLflow converter factory.

    Args:
        model_framework (str): e.g., ModelFramework.HUGGINGFACE.value.
        canonical_task (str): A canonical task identifier (e.g., 'nlp', 'vision').
        model_family (str, optional): An additional identifier if needed.
    """
    def decorator(cls):
        key = (model_framework, canonical_task, model_family)
        if key in MLFLOW_CONVERTER_REGISTRY:
            raise Exception(f"Converter for key {key} is already registered!")
        MLFLOW_CONVERTER_REGISTRY[key] = cls
        return cls
    return decorator

def get_canonical_key(model_framework, translate_params):
    """
    Returns a canonical key tuple (model_framework, canonical_task, model_family)
    based on the raw parameters. For Hugging Face, it uses external rule logic.
    """
    task = translate_params["task"]
    model_id = translate_params.get("model_id", "")
    model_family = None
    canonical_task = task

    if model_framework == ModelFramework.HUGGINGFACE.value:
        canonical_task, model_family = get_huggingface_rule(task, model_id, translate_params)
    # Additional framework-specific logic can be added here.
    # e.g., for MMLAB, LLAVA, AutoML, etc.

    return (model_framework, canonical_task, model_family)

def get_mlflow_converter(model_framework, model_dir, output_dir, temp_dir, translate_params):
    """
    Looks up the appropriate factory in the registry using the canonical key and returns
    an instantiated MLflow converter.
    """
    key = get_canonical_key(model_framework, translate_params)
    logger.info(f"Looking up converter with key: {key}")
    factory_cls = MLFLOW_CONVERTER_REGISTRY.get(key)
    if not factory_cls:
        # Fallback: ignore model_family if not found.
        fallback_key = (model_framework, key[1], None)
        factory_cls = MLFLOW_CONVERTER_REGISTRY.get(fallback_key)
    if not factory_cls:
        raise Exception(f"No converter registered for key {key}")
    return factory_cls.create_mlflow_converter(model_dir, output_dir, temp_dir, translate_params)

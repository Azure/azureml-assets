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
    VirchowMLFlowConvertor,
    HibouBMLFlowConvertor,
)


logger = get_logger(__name__)


def get_mlflow_convertor(model_framework, model_dir, output_dir, temp_dir, translate_params):
    """Instantiate and return MLflow convertor."""
    task = translate_params["task"]
    model_id = translate_params.get("model_id")
    logger.info(f"Model task:{task}, Model ID:{model_id}")
    if model_id is None:
        model_id = ""

    if model_framework == ModelFramework.HUGGINGFACE.value:
        # Models from Hugging face framework exported in transformers mlflow flavor
        if SupportedNLPTasks.has_value(task):
            return NLPMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        elif SupportedVisionTasks.has_value(task):
            return VisionMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif task in [PyFuncSupportedTasks.TEXT_TO_IMAGE.value,
                      PyFuncSupportedTasks.TEXT_TO_IMAGE_INPAINTING.value,
                      PyFuncSupportedTasks.IMAGE_TO_IMAGE.value]:
            return TextToImageMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif task == SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value:
            return ASRMLflowConvertorFactory.create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        # Models from Hugging face framework exported in PyFunc mlflow flavor
        elif (task == PyFuncSupportedTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value) or (
                (task == PyFuncSupportedTasks.EMBEDDINGS.value) and model_id.startswith(ModelFamilyPrefixes.CLIP.value)
        ):
            return CLIPMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif (task == PyFuncSupportedTasks.EMBEDDINGS.value) and model_id.startswith(ModelFamilyPrefixes.DINOV2.value):
            return DinoV2MLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif task in [PyFuncSupportedTasks.IMAGE_TO_TEXT.value, PyFuncSupportedTasks.VISUAL_QUESTION_ANSWERING.value]:
            return BLIPMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif task == PyFuncSupportedTasks.MASK_GENERATION.value:
            return SegmentAnythingMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif (
            task == PyFuncSupportedTasks.FEATURE_EXTRACTION.value
            or task == PyFuncSupportedTasks.IMAGE_FEATURE_EXTRACTION.value
        ) and model_id.startswith(ModelFamilyPrefixes.HIBOU_B.value):
            return HibouBMLFlowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif task == PyFuncSupportedTasks.IMAGE_FEATURE_EXTRACTION.value:
            return VirchowMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        else:
            raise Exception(
                f"Models from {model_framework} for task {task} and model {model_id} "
                "not supported for MLflow conversion"
            )
    elif model_framework == ModelFramework.MMLAB.value:
        # Models from MMLAB model framework exported in PyFunc mlflow flavor
        if MMLabDetectionTasks.has_value(task):
            return MMLabDetectionMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        elif MMLabTrackingTasks.has_value(task):
            return MMLabTrackingMLflowConvertorFactory.create_mlflow_convertor(
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
    elif model_framework == ModelFramework.AutoML.value:
        # Models from AutML model framework exported in PyFunc mlflow flavor
        if task in [
            PyFuncSupportedTasks.IMAGE_CLASSIFICATION.value,
            PyFuncSupportedTasks.IMAGE_CLASSIFICATION_MULTILABEL.value,
            PyFuncSupportedTasks.IMAGE_OBJECT_DETECTION.value,
            PyFuncSupportedTasks.IMAGE_INSTANCE_SEGMENTATION.value,
        ]:
            return AutoMLMLflowConvertorFactory.create_mlflow_convertor(
                model_dir, output_dir, temp_dir, translate_params
            )
        else:
            raise Exception(
                f"Models from {model_framework} for {task} not supported for MLflow conversion"
            )
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


class TextToImageMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for text to image model."""

    STABLE_DIFFUSION_TASK_MAP = {
        PyFuncSupportedTasks.TEXT_TO_IMAGE.value: StableDiffusionMlflowConvertor,
        PyFuncSupportedTasks.TEXT_TO_IMAGE_INPAINTING.value: StableDiffusionInpaintingMlflowConvertor,
        PyFuncSupportedTasks.IMAGE_TO_IMAGE.value: StableDiffusionImageToImageMlflowConvertor,

    }

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for diffusers."""
        misc = translate_params["misc"]
        kwargs = {}
        model_family = None

        def get_text_to_image_model_family():
            if not misc:
                return None
            for model_family in SupportedTextToImageModelFamily.list_values():
                if model_family in misc:
                    return model_family
            return None

        if translate_params["task"] == PyFuncSupportedTasks.TEXT_TO_IMAGE.value:
            model_family = get_text_to_image_model_family()
            if not model_family:
                raise Exception("Unsupported model family for text to image model")
            kwargs.update({"model_family": model_family})
        try:
            converter = TextToImageMLflowConvertorFactory.STABLE_DIFFUSION_TASK_MAP[translate_params["task"]]
            return converter(
                model_dir=model_dir,
                output_dir=output_dir,
                temp_dir=temp_dir,
                translate_params=translate_params,
                **kwargs
            )
        except KeyError:
            raise Exception("Unsupported task for stable diffusion model family")


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


class BLIPMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for BLIP model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for BLIP model family."""
        return BLIPMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class DinoV2MLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for DinoV2 model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for DinoV2 model."""
        return DinoV2MLFlowConvertor(
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


class SegmentAnythingMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for segment anything (SAM) model."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for segment anything (SAM) model."""
        return SegmentAnythingMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class VirchowMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for segment anything Virchow model."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for segment anything Virchow model."""
        return VirchowMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class HibouBMLFlowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for segment anything HibouB model."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for segment anything HibouB model."""
        return HibouBMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class MMLabTrackingMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for MMTrack video model family."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for vision tasks."""
        return MMLabTrackingMLflowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )


class AutoMLMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for AutoML models."""

    def create_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for AutoML models."""
        return AutoMLMLFlowConvertor(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )

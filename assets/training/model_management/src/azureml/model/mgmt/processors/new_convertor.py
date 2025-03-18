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

from abc import ABC, abstractmethod

class MLflowConvertorFactoryInterface(ABC):
    """MLflow convertor factory interface."""

    @abstractmethod
    def create_mlflow_convertor(self, model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor."""
        raise NotImplementedError

class BaseMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Base factory class for MLflow convertors."""

    CONVERTOR_CLASSES = {
        "NLP": NLPMLflowConvertor,
        "Vision": VisionMLflowConvertor,
        "MMLabDetection": MMLabDetectionMLflowConvertor,
        "CLIP": CLIPMLFlowConvertor,
        "BLIP": BLIPMLFlowConvertor,
        "DinoV2": DinoV2MLFlowConvertor,
        "LLaVA": LLaVAMLFlowConvertor,
        "SegmentAnything": SegmentAnythingMLFlowConvertor,
        "Virchow": VirchowMLFlowConvertor,
        "MMLabTracking": MMLabTrackingMLflowConvertor,
        "AutoML": AutoMLMLFlowConvertor,
    }

    def __init__(self, convertor_type):
        if convertor_type not in self.CONVERTOR_CLASSES:
            raise ValueError(f"Unsupported convertor type: {convertor_type}")
        self.convertor_class = self.CONVERTOR_CLASSES[convertor_type]

    def create_mlflow_convertor(self, model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor with common initialization."""
        return self.convertor_class(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )

# Specialized Factories with Custom Covertor Logic

class ASRMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for ASR model family."""
    
    ASR_MODELS = {
        SupportedASRModelFamily.WHISPER.value: WhisperMLflowConvertor
    }

    def create_mlflow_convertor(self, model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for ASR tasks."""
        misc = translate_params.get("misc", [])
        model_family = next((m for m in self.ASR_MODELS if m in misc), None)
        if not model_family:
            raise Exception("Unsupported ASR model family")
        return self.ASR_MODELS[model_family](
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
        )

class TextToImageMLflowConvertorFactory(MLflowConvertorFactoryInterface):
    """Factory class for text to image model."""

    STABLE_DIFFUSION_TASK_MAP = {
        PyFuncSupportedTasks.TEXT_TO_IMAGE.value: StableDiffusionMlflowConvertor,
        PyFuncSupportedTasks.TEXT_TO_IMAGE_INPAINTING.value: StableDiffusionInpaintingMlflowConvertor,
        PyFuncSupportedTasks.IMAGE_TO_IMAGE.value: StableDiffusionImageToImageMlflowConvertor,
    }

    def create_mlflow_convertor(self, model_dir, output_dir, temp_dir, translate_params):
        """Create MLflow convertor for diffusers."""
        task = translate_params.get("task")
        misc = translate_params.get("misc", [])
        kwargs = {}

        if task == PyFuncSupportedTasks.TEXT_TO_IMAGE.value:
            model_family = next((m for m in SupportedTextToImageModelFamily.list_values() if m in misc), None)
            if not model_family:
                raise Exception("Unsupported model family for text to image model")
            kwargs["model_family"] = model_family
        
        converter_cls = self.STABLE_DIFFUSION_TASK_MAP.get(task)
        if not converter_cls:
            raise Exception("Unsupported task for stable diffusion model family")
        
        return converter_cls(
            model_dir=model_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            translate_params=translate_params,
            **kwargs
        )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyFunc MLflow model convertors."""

import json
import mlflow
import os
import sys
import torch

from abc import ABC, abstractmethod
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, ParamSchema, ParamSpec
from pathlib import Path
from typing import Dict, List, Optional

from azureml.model.mgmt.utils.common_utils import fetch_mlflow_acft_metadata
from azureml.model.mgmt.utils.logging_utils import get_logger
from azureml.model.mgmt.processors.convertors import MLFLowConvertorInterface
from azureml.model.mgmt.processors.pyfunc.config import (
    MMLabDetectionTasks, MMLabTrackingTasks, SupportedTasks)

from azureml.model.mgmt.processors.pyfunc.clip.config import \
    MLflowSchemaLiterals as CLIPMLFlowSchemaLiterals, MLflowLiterals as CLIPMLflowLiterals, \
    Tasks as CLIPTasks
from azureml.model.mgmt.processors.pyfunc.dinov2.config import \
    MLflowSchemaLiterals as DinoV2MLFlowSchemaLiterals, MLflowLiterals as DinoV2MLflowLiterals, \
    Tasks as DinoV2Tasks
from azureml.model.mgmt.processors.pyfunc.blip.config import \
    MLflowSchemaLiterals as BLIPMLFlowSchemaLiterals, MLflowLiterals as BLIPMLflowLiterals
from azureml.model.mgmt.processors.pyfunc.text_to_image.config import (
    MLflowSchemaLiterals as TextToImageMLFlowSchemaLiterals,
    MLflowLiterals as TextToImageMLflowLiterals,
)
from azureml.model.mgmt.processors.pyfunc.llava.config import \
    MLflowSchemaLiterals as LLaVAMLFlowSchemaLiterals, MLflowLiterals as LLaVAMLflowLiterals
from azureml.model.mgmt.processors.pyfunc.segment_anything.config import \
    MLflowSchemaLiterals as SegmentAnythingMLFlowSchemaLiterals, MLflowLiterals as SegmentAnythingMLflowLiterals
from azureml.model.mgmt.processors.pyfunc.vision.config import \
    MLflowSchemaLiterals as VisionMLFlowSchemaLiterals, MMDetLiterals
from azureml.model.mgmt.processors.pyfunc.virchow.config import \
    MLflowSchemaLiterals as VirchowMLFlowSchemaLiterals, MLflowLiterals as VirchowMLflowLiterals
from azureml.model.mgmt.processors.pyfunc.hibou_b.config import \
    MLflowSchemaLiterals as HibouBMLFlowSchemaLiterals, MLflowLiterals as HibouBMLflowLiterals


logger = get_logger(__name__)


class PyFuncMLFLowConvertor(MLFLowConvertorInterface, ABC):
    """PyFunc MLflow convertor base class."""

    CONDA_FILE_NAME = "conda.yaml"
    REQUIREMENTS_FILE_NAME = "requirements.txt"
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")
    sys.path.append(COMMON_DIR)

    @abstractmethod
    def get_model_signature(self) -> ModelSignature:
        """Return model signature for MLflow model.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        raise NotImplementedError

    @abstractmethod
    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        raise NotImplementedError

    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        temp_dir: Path,
        translate_params: Dict,
    ):
        """Initialize MLflow convertor for PyFunc models."""
        self._validate(translate_params)
        self._model_dir = os.fspath(model_dir)
        self._output_dir = output_dir
        self._temp_dir = temp_dir
        self._model_id = translate_params.get("model_id")
        self._task = translate_params["task"]
        self._signatures = translate_params.get("signatures", None)
        self._inference_base_image = translate_params.get("inference_base_image", None)

    def _save(
        self,
        mlflow_model_wrapper: PyFuncModel,
        artifacts_dict: Dict[str, str],
        code_path: List[str],
        pip_requirements: Optional[str] = None,
        conda_env: Optional[str] = None,
        metadata: Optional[Dict] = {}
    ):
        """Save Mlflow model to output directory.

        :param mlflow_model_wrapper: MLflow model wrapper instance
        :type mlflow_model_wrapper: Subclass of PyFuncModel
        :param artifacts_dict: Dictionary of name to artifact path
        :type artifacts_dict: Dict[str, str]
        :param pip_requirements: Path to pip requirements file
        :type pip_requirements: Optional[str]
        :param conda_env: Path to conda environment yaml file
        :type conda_env: Optional[str]
        :param code_path: A list of local filesystem paths to Python file dependencies
        :type code_path: List[str]
        :param metadata: A metadata dictionary to associate with the MLflow model
        :type metadata: Optional[Dict]. Defaults to {}.
        """
        signatures = self._signatures or self.get_model_signature()
        # set metadata info Check
        metadata.update(fetch_mlflow_acft_metadata(
            base_model_name=self._model_id,
            is_finetuned_model=False,
            base_model_task=self._task
        ))
        mlflow.pyfunc.save_model(
            path=self._output_dir,
            python_model=mlflow_model_wrapper,
            artifacts=artifacts_dict,
            pip_requirements=pip_requirements,
            conda_env=conda_env,
            signature=signatures,
            code_path=code_path,
            metadata=metadata,
        )

        logger.info("Model saved successfully.")

    def _validate(self, translate_params):
        """Validate translate parameters."""
        if not translate_params.get("task"):
            raise Exception("task is a required parameter for pyfunc flavor.")
        task = translate_params["task"]
        if not SupportedTasks.has_value(task):
            raise Exception(f"Unsupported task {task} for pyfunc flavor.")


class MMLabDetectionMLflowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for detection models from MMLab."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "vision")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for vision models."""
        super().__init__(**kwargs)
        if not MMLabDetectionTasks.has_value(self._task):
            raise Exception("Unsupported vision task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(VisionMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE,
                        VisionMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE)
            ]
        )

        if self._task in [MMLabDetectionTasks.MM_OBJECT_DETECTION.value,
                          MMLabDetectionTasks.MM_INSTANCE_SEGMENTATION.value]:
            output_schema = Schema(
                [
                    ColSpec(VisionMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            VisionMLFlowSchemaLiterals.OUTPUT_COLUMN_BOXES),
                ]
            )
            param_schema = ParamSchema(
                [
                    ParamSpec(MMDetLiterals.TEXT_PROMPT,
                              DataType.string, None),
                    ParamSpec(MMDetLiterals.CUSTOM_ENTITIES,
                              DataType.boolean, True),
                ]
            )
        else:
            raise NotImplementedError(f"Task type: {self._task} is not supported yet.")
        return ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from detection_predict import ImagesDetectionMLflowModelWrapper

        mlflow_model_wrapper = ImagesDetectionMLflowModelWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        if self._task == MMLabDetectionTasks.MM_OBJECT_DETECTION.value:
            pip_requirements = os.path.join(self.MODEL_DIR, "mmdet-od-requirements.txt")
        elif self._task == MMLabDetectionTasks.MM_INSTANCE_SEGMENTATION.value:
            pip_requirements = os.path.join(self.MODEL_DIR, "mmdet-is-requirements.txt")
        else:
            pip_requirements = None
        code_path = [
            os.path.join(self.MODEL_DIR, "detection_predict.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]
        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            pip_requirements=pip_requirements,
            code_path=code_path,
        )

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        metadata_path = os.path.join(self._model_dir, "model_selector_args.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        artifacts_dict = {
            MMDetLiterals.CONFIG_PATH: os.path.join(self._model_dir, metadata.get("pytorch_model_path")),
            MMDetLiterals.WEIGHTS_PATH: os.path.join(self._model_dir, metadata.get("model_weights_path_or_url")),
            MMDetLiterals.METAFILE_PATH: os.path.join(self._model_dir, metadata.get("model_metafile_path")),
        }
        return artifacts_dict


class CLIPMLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for CLIP models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "clip")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for CLIP models."""
        super().__init__(**kwargs)
        if self._task not in \
                [SupportedTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value, SupportedTasks.EMBEDDINGS.value]:
            raise Exception("Unsupported task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(CLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE,
                        CLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE),
                ColSpec(CLIPMLFlowSchemaLiterals.INPUT_COLUMN_TEXT_DATA_TYPE,
                        CLIPMLFlowSchemaLiterals.INPUT_COLUMN_TEXT),
            ]
        )

        if self._task == SupportedTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value:
            output_schema = Schema(
                [
                    ColSpec(CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS),
                    ColSpec(CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS),
                ]
            )
        elif self._task == SupportedTasks.EMBEDDINGS.value:
            output_schema = Schema(
                [
                    ColSpec(CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE_FEATURES),
                    ColSpec(CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_TEXT_FEATURES),
                ]
            )
        else:
            raise Exception("Unsupported task")

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)

        if self._task == SupportedTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value:
            from clip_mlflow_wrapper import CLIPMLFlowModelWrapper
            mlflow_model_wrapper = CLIPMLFlowModelWrapper(task_type=CLIPTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value)
        elif self._task == SupportedTasks.EMBEDDINGS.value:
            from clip_embeddings_mlflow_wrapper import CLIPEmbeddingsMLFlowModelWrapper
            mlflow_model_wrapper = CLIPEmbeddingsMLFlowModelWrapper(task_type=CLIPTasks.EMBEDDINGS.value)
        else:
            raise Exception("Unsupported task")

        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = self._get_code_path()

        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )

    def _get_code_path(self):
        """Return code path for saving mlflow model depending on task type.

        :return: code path
        :rtype: List[str]
        """
        code_path = [
            os.path.join(self.MODEL_DIR, "clip_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]
        if self._task == SupportedTasks.EMBEDDINGS.value:
            code_path.append(os.path.join(self.MODEL_DIR, "clip_embeddings_mlflow_wrapper.py"))

        return code_path

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        artifacts_dict = {
            CLIPMLflowLiterals.MODEL_DIR: self._model_dir
        }
        return artifacts_dict


class DinoV2MLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for DinoV2 models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "dinov2")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for DinoV2 models."""
        super().__init__(**kwargs)

        if self._task != SupportedTasks.EMBEDDINGS.value:
            raise Exception("Unsupported task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(DataType.string, DinoV2MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE),
            ]
        )

        output_schema = Schema(
            [
                ColSpec(DataType.string, DinoV2MLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE_FEATURES),
            ]
        )

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)

        from dinov2_embeddings_mlflow_wrapper import DinoV2EmbeddingsMLFlowModelWrapper
        mlflow_model_wrapper = DinoV2EmbeddingsMLFlowModelWrapper(task_type=DinoV2Tasks.EMBEDDINGS.value)

        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = self._get_code_path()

        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )

    def _get_code_path(self):
        """Return code path for saving mlflow model depending on task type.

        :return: code path
        :rtype: List[str]
        """
        code_path = [
            os.path.join(self.MODEL_DIR, "dinov2_embeddings_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]

        return code_path

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        artifacts_dict = {
            DinoV2MLflowLiterals.MODEL_DIR: self._model_dir
        }
        return artifacts_dict


class BLIPMLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for BLIP models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "blip")
    COMMON_DIR = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for BLIP models."""
        super().__init__(**kwargs)
        if self._task not in [SupportedTasks.IMAGE_TO_TEXT.value, SupportedTasks.VISUAL_QUESTION_ANSWERING.value]:
            raise Exception("Unsupported task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        if self._task == SupportedTasks.IMAGE_TO_TEXT.value:
            input_schema = Schema(
                [
                    ColSpec(BLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE,
                            BLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE),
                ]
            )
        elif self._task == SupportedTasks.VISUAL_QUESTION_ANSWERING.value:
            input_schema = Schema(
                [
                    ColSpec(BLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE,
                            BLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE),
                    ColSpec(BLIPMLFlowSchemaLiterals.INPUT_COLUMN_TEXT_DATA_TYPE,
                            BLIPMLFlowSchemaLiterals.INPUT_COLUMN_TEXT),
                ]
            )
        else:
            raise Exception("Unsupported task")

        output_schema = Schema(
            [
                ColSpec(BLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                        BLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_TEXT),
            ]
        )

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from mlflow_wrapper import BLIPMLFlowModelWrapper

        mlflow_model_wrapper = BLIPMLFlowModelWrapper(task_type=self._task, model_id=self._model_id)
        artifacts_dict = {
            BLIPMLflowLiterals.MODEL_DIR: self._model_dir
        }
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = [
            os.path.join(self.MODEL_DIR, "mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py"),
        ]
        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )


class TextToImageMLflowConvertor(PyFuncMLFLowConvertor):
    """MlfLow convertor base class for text to image models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "text_to_image")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for text to image models."""
        self._model_family = kwargs.pop("model_family", None)
        super().__init__(**kwargs)

    def get_model_signature(self):
        """Return model signature for text to image models."""
        input_schema = Schema(inputs=[
            ColSpec(name=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT,
                    type=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT_DATA_TYPE,)
        ])
        output_schema = Schema(inputs=[
            ColSpec(name=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT,
                    type=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT_DATA_TYPE,),
            ColSpec(name=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE,
                    type=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE_TYPE),
            ColSpec(name=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG,
                    type=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG_TYPE,),
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)


class StableDiffusionMlflowConvertor(TextToImageMLflowConvertor):
    """HF MlfLow convertor class for stable diffusion models."""

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for SD models."""
        super().__init__(**kwargs)

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        artifacts_dict = {
            TextToImageMLflowLiterals.MODEL_DIR: self._model_dir
        }
        return artifacts_dict

    def save_as_mlflow(self):
        """Prepare SD model for save to MLflow."""
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from stable_diffusion_mlflow_wrapper import StableDiffusionMLflowWrapper

        mlflow_model_wrapper = StableDiffusionMLflowWrapper(task_type=self._task, model_family=self._model_family)
        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = [
            os.path.join(self.MODEL_DIR, "stable_diffusion_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]

        metadata = {"model_type": "stable-diffusion"}
        if self._inference_base_image:
            metadata["azureml.base_image"] = self._inference_base_image

        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
            metadata=metadata
        )


class TextToImageInpaintingMLflowConvertor(PyFuncMLFLowConvertor):
    """MlfLow convertor base class for text to image inpainting models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "text_to_image")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for text to image models."""
        super().__init__(**kwargs)

    def get_model_signature(self):
        """Return model signature for text to image models."""
        input_schema = Schema(inputs=[
            ColSpec(name=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT,
                    type=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT_DATA_TYPE,),
            ColSpec(name=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE,
                    type=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_TYPE,),
            ColSpec(name=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_MASK_IMAGE,
                    type=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_MASK_IMAGE_TYPE,)
        ])
        output_schema = Schema(inputs=[
            ColSpec(name=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE,
                    type=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE_TYPE),
            ColSpec(
                name=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG,
                type=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG_TYPE,
            ),
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)


class StableDiffusionInpaintingMlflowConvertor(TextToImageInpaintingMLflowConvertor):
    """HF MlfLow convertor class for stable diffusion inpainting models."""

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for SD inpainting models."""
        super().__init__(**kwargs)

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        artifacts_dict = {
            TextToImageMLflowLiterals.MODEL_DIR: self._model_dir
        }
        return artifacts_dict

    def save_as_mlflow(self):
        """Prepare SD model for save to MLflow."""
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from stable_diffusion_inpainting_mlflow_wrapper import StableDiffusionInpaintingMLflowWrapper

        mlflow_model_wrapper = StableDiffusionInpaintingMLflowWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = [
            os.path.join(self.MODEL_DIR, "stable_diffusion_inpainting_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]

        # Enable DS-MII optimisation for Stable Diffusion Inpainting
        metadata = {"model_type": "stable-diffusion"}
        if self._inference_base_image:
            metadata["azureml.base_image"] = self._inference_base_image

        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )


class ImageTextToImageMLflowConvertor(PyFuncMLFLowConvertor):
    """MlfLow convertor base class for image-text to image models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "text_to_image")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for image-text to image models."""
        super().__init__(**kwargs)

    def get_model_signature(self):
        """Return model signature for image-text to image models."""
        input_schema = Schema(inputs=[
            ColSpec(name=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT,
                    type=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT_DATA_TYPE),
            ColSpec(name=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE,
                    type=TextToImageMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_TYPE)
        ])
        output_schema = Schema(inputs=[
            ColSpec(name=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE,
                    type=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE_TYPE),
            ColSpec(name=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG,
                    type=TextToImageMLFlowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG_TYPE),
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)


class StableDiffusionImageToImageMlflowConvertor(ImageTextToImageMLflowConvertor):
    """HF MlfLow convertor class for stable diffusion image-text to image models."""

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for SD image-text to image models."""
        super().__init__(**kwargs)

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        artifacts_dict = {
            TextToImageMLflowLiterals.MODEL_DIR: self._model_dir
        }
        return artifacts_dict

    def save_as_mlflow(self):
        """Prepare SD model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from stable_diffusion_image_to_image_mlflow_wrapper import StableDiffusionImageTexttoImageMLflowWrapper

        mlflow_model_wrapper = StableDiffusionImageTexttoImageMLflowWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = [
            os.path.join(self.MODEL_DIR, "stable_diffusion_image_to_image_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]
        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )


class LLaVAMLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for LLaVA models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "llava")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for LLaVA models."""
        super().__init__(**kwargs)
        if self._task != SupportedTasks.IMAGE_TEXT_TO_TEXT.value:
            raise Exception("Unsupported task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(DataType.string, LLaVAMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE),
                ColSpec(DataType.string, LLaVAMLFlowSchemaLiterals.INPUT_COLUMN_PROMPT),
                ColSpec(DataType.string, LLaVAMLFlowSchemaLiterals.INPUT_COLUMN_DIRECT_QUESTION),
            ]
        )

        output_schema = Schema(
            [
                ColSpec(DataType.string, LLaVAMLFlowSchemaLiterals.OUTPUT_COLUMN_RESPONSE)
            ]
        )

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from llava_mlflow_wrapper import LLaVAMLflowWrapper

        mlflow_model_wrapper = LLaVAMLflowWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = [
            os.path.join(self.MODEL_DIR, "llava_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]
        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        # Get the name of the only subdirectory of the model directory.
        sd = next(
            (d for d in os.listdir(self._model_dir) if os.path.isdir(os.path.join(self._model_dir, d))),
            self._model_dir
        )

        # Set model_dir parameter to point to subdirectory.
        artifacts_dict = {
            LLaVAMLflowLiterals.MODEL_DIR: os.path.join(self._model_dir, sd)
        }
        return artifacts_dict


class MMLabTrackingMLflowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for tracking models from MMLab."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "vision")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for vision models."""
        super().__init__(**kwargs)
        if not MMLabTrackingTasks.has_value(self._task):
            raise Exception("Unsupported vision task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(VisionMLFlowSchemaLiterals.INPUT_COLUMN_VIDEO_DATA_TYPE,
                        VisionMLFlowSchemaLiterals.INPUT_COLUMN_VIDEO)
            ]
        )

        if self._task in [MMLabTrackingTasks.MM_MULTI_OBJECT_TRACKING.value]:
            output_schema = Schema(
                [
                    ColSpec(VisionMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            VisionMLFlowSchemaLiterals.OUTPUT_COLUMN_BOXES),
                ]
            )
        else:
            raise NotImplementedError(f"Task type: {self._task} is not supported yet.")
        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from track_predict import VideosTrackingMLflowModelWrapper

        mlflow_model_wrapper = VideosTrackingMLflowModelWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        conda_env = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = [
            os.path.join(self.MODEL_DIR, "track_predict.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]
        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env,
            code_path=code_path,
        )

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        metadata_path = os.path.join(self._model_dir, "model_selector_args.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        artifacts_dict = {
            MMDetLiterals.CONFIG_PATH: os.path.join(self._model_dir, metadata.get("pytorch_model_path")),
            MMDetLiterals.WEIGHTS_PATH: os.path.join(self._model_dir, metadata.get("model_weights_path_or_url")),
            MMDetLiterals.METAFILE_PATH: os.path.join(self._model_dir, metadata.get("model_metafile_path")),
        }
        return artifacts_dict


class SegmentAnythingMLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for SAM models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "segment_anything")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for SAM models."""
        super().__init__(**kwargs)
        if self._task != SupportedTasks.MASK_GENERATION.value:
            raise Exception("Unsupported task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE,
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE,
                ),
                ColSpec(
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_INPUT_POINTS_DATA_TYPE,
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_INPUT_POINTS,
                ),
                ColSpec(
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_INPUT_BOXES_DATA_TYPE,
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_INPUT_BOXES,
                ),
                ColSpec(
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_INPUT_LABELS_DATA_TYPE,
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_COLUMN_INPUT_LABELS,
                ),
                ColSpec(
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_PARAM_MULTIMASK_OUTPUT_DATA_TYPE,
                    SegmentAnythingMLFlowSchemaLiterals.INPUT_PARAM_MULTIMASK_OUTPUT,
                ),
            ]
        )

        output_schema = Schema(
            [
                ColSpec(
                    SegmentAnythingMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                    SegmentAnythingMLFlowSchemaLiterals.OUTPUT_COLUMN_RESPONSE,
                )
            ]
        )

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from segment_anything_mlflow_wrapper import SegmentAnythingMLflowWrapper

        mlflow_model_wrapper = SegmentAnythingMLflowWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = [
            os.path.join(self.MODEL_DIR, "segment_anything_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py"),
        ]
        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        artifacts_dict = {SegmentAnythingMLflowLiterals.MODEL_DIR: self._model_dir}
        return artifacts_dict


class AutoMLMLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for AutoML models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "automl")
    MLflowLiteral_Model = "model"
    MLflowLiteral_Settings = "settings"

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for AutoML models."""
        super().__init__(**kwargs)
        if self._task not in [
            SupportedTasks.IMAGE_CLASSIFICATION.value,
            SupportedTasks.IMAGE_CLASSIFICATION_MULTILABEL.value,
            SupportedTasks.IMAGE_OBJECT_DETECTION.value,
            SupportedTasks.IMAGE_INSTANCE_SEGMENTATION.value,
        ]:
            raise Exception("Unsupported task")

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _get_model_weights_classification(self) -> str:
        """Return default model weights path.

        :return: Path to default model.
        :rtype: str
        """
        multilabel = False
        if self._task == SupportedTasks.IMAGE_CLASSIFICATION_MULTILABEL.value:
            multilabel = True

        from azureml.automl.dnn.vision.classification.models import ModelFactory
        from azureml.automl.dnn.vision.classification.common.constants import ModelNames
        from azureml.automl.dnn.vision.common.constants import (
            PretrainedModelUrls,
            PretrainedModelNames,
        )
        from azureml.automl.dnn.vision.common.pretrained_model_utilities import (
            PretrainedModelFactory,
        )
        import copy

        with open(os.path.join(self.MODEL_DIR, "vit_classes.txt")) as f:
            vit_classes = f.readlines()

        model_wrapper = ModelFactory().get_model_wrapper(
            model_name=ModelNames.VITB16R224,
            num_classes=len(vit_classes),
            multilabel=multilabel,
            distributed=False,
            local_rank=0,
            device=self._device,
            model_state=PretrainedModelFactory._load_state_dict_from_url_with_retry(
                PretrainedModelUrls.MODEL_URLS[PretrainedModelNames.VITB16R224],
                progress=True,
            ),
            settings={},
        )

        specs = {
            "multilabel": model_wrapper.multilabel,
            "model_settings": model_wrapper.model_settings,
            "labels": vit_classes,
        }

        checkpoint_data = {
            "model_name": model_wrapper.model_name,
            "number_of_classes": model_wrapper.number_of_classes,
            "specs": specs,
            "model_state": copy.deepcopy(model_wrapper.state_dict()),
        }

        model_file = "/tmp/vitb16r224-3c68ea1f.pth"
        torch.save(checkpoint_data, model_file)

        return model_file

    def _get_model_weights_object_detection(self) -> str:
        """Return default model weights path.

        :return: Path to default model.
        :rtype: str
        """
        from azureml.automl.dnn.vision.object_detection.models.detection import setup_model
        from azureml.automl.dnn.vision.object_detection.common.constants import ModelNames
        import copy
        with open(os.path.join(self.MODEL_DIR, "coco_od_classes.txt")) as f:
            coco_classes = f.readlines()
        model_wrapper = setup_model(
            model_name=ModelNames.YOLO_V5,
            number_of_classes=len(coco_classes),
            classes=coco_classes,
            device=self._device,
            distributed=False,
            local_rank=0,
            model_state=None,
            specs={"model_size": "medium",
                   "device": self._device,
                   "img_size": 640,
                   "box_score_thresh": 0.1,
                   "nms_iou_thresh": 0.5},
            settings={})

        specs = {
            "model_specs": model_wrapper.specs,
            "model_settings": model_wrapper.model_settings.get_settings_dict(),
            "inference_settings": model_wrapper.inference_settings,
            'classes': model_wrapper.classes,
        }

        checkpoint_data = {
            "model_name": model_wrapper.model_name,
            "number_of_classes": model_wrapper.number_of_classes,
            "specs": specs,
            "model_state": copy.deepcopy(model_wrapper.state_dict()),
        }

        model_file = "/tmp/yolov5.pth"
        torch.save(checkpoint_data, model_file)

        return model_file

    def _get_model_weights_instance_segmentation(self) -> str:
        """Return default model weights path.

        :return: Path to default model.
        :rtype: str
        """
        from azureml.automl.dnn.vision.object_detection.models.detection import setup_model
        from azureml.automl.dnn.vision.object_detection.common.constants import ModelNames
        from azureml.automl.dnn.vision.common.pretrained_model_utilities import load_state_dict_from_url
        from azureml.automl.dnn.vision.common.constants import (
            PretrainedModelUrls,
            PretrainedModelNames,
        )
        with open(os.path.join(self.MODEL_DIR, "coco_seg_classes.txt")) as f:
            coco_classes = f.readlines()
        model_wrapper = setup_model(
            model_name=ModelNames.MASK_RCNN_RESNET50_FPN,
            number_of_classes=len(coco_classes),
            classes=coco_classes,
            device=self._device,
            distributed=False,
            local_rank=0,
            model_state=None,
            specs=None,
            settings={})

        specs = {
            "model_specs": model_wrapper.specs,
            "model_settings": model_wrapper.model_settings.get_settings_dict(),
            "inference_settings": model_wrapper.inference_settings,
            'classes': model_wrapper.classes,
        }

        # ideally, model_state_dict should be copy.deepcopy(model_wrapper.state_dict())
        # for instance segmentation, when the pretrained weight is loaded in dnn-vision package,
        # it would randomly initialize the predictor head (regardless of number_of_classes is the same as default)
        # thus we need to use the temp workaround to load the pretrained weight from url

        model_url = PretrainedModelUrls.MODEL_URLS[PretrainedModelNames.MASKRCNN_RESNET50_FPN_COCO]
        model_state_dict = load_state_dict_from_url(model_url)
        checkpoint_data = {
            "model_name": model_wrapper.model_name,
            "number_of_classes": model_wrapper.number_of_classes,
            "specs": specs,
            "model_state": model_state_dict,
        }

        model_file = "/tmp/maskrcnn.pth"
        torch.save(checkpoint_data, model_file)

        return model_file

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        from azureml.automl.dnn.vision.common.model_export_utils import _get_mlflow_signature
        return _get_mlflow_signature(self._task)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        from azureml.automl.dnn.vision.common.mlflow.mlflow_model_wrapper import MLFlowImagesModelWrapper
        from azureml.automl.dnn.vision.common.model_export_utils import _get_scoring_method
        is_yolo = self._task == SupportedTasks.IMAGE_OBJECT_DETECTION.value
        mlflow_model_wrapper = MLFlowImagesModelWrapper(model_settings={},
                                                        task_type=self._task,
                                                        scoring_method=_get_scoring_method(self._task, is_yolo))

        if self._task in [
            SupportedTasks.IMAGE_CLASSIFICATION.value,
            SupportedTasks.IMAGE_CLASSIFICATION_MULTILABEL.value,
        ]:
            model_file = self._get_model_weights_classification()
        elif self._task == SupportedTasks.IMAGE_OBJECT_DETECTION.value:
            model_file = self._get_model_weights_object_detection()
        elif self._task == SupportedTasks.IMAGE_INSTANCE_SEGMENTATION.value:
            model_file = self._get_model_weights_instance_segmentation()
        else:
            raise Exception("Unsupported task")

        artifacts_dict = {
            self.MLflowLiteral_Model: model_file,
            self.MLflowLiteral_Settings: os.path.join(self.MODEL_DIR, "settings.json"),
        }
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")

        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=None,
        )


class VirchowMLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for Virchow models."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "virchow")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for Virchow models."""
        super().__init__(**kwargs)
        if self._task not in \
                [SupportedTasks.IMAGE_FEATURE_EXTRACTION.value]:
            raise Exception("Unsupported task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(VirchowMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE,
                        VirchowMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE),
                ColSpec(VirchowMLFlowSchemaLiterals.INPUT_COLUMN_TEXT_DATA_TYPE,
                        VirchowMLFlowSchemaLiterals.INPUT_COLUMN_TEXT),
            ]
        )
        params = ParamSchema(
                [
                    ParamSpec(VirchowMLflowLiterals.DEVICE_TYPE,
                              DataType.string, "cuda"),
                    ParamSpec(VirchowMLflowLiterals.TO_HALF_PRECISION,
                              DataType.boolean, False),
                ]
            )
        if self._task == SupportedTasks.IMAGE_FEATURE_EXTRACTION.value:
            output_schema = Schema(
                [
                    ColSpec(VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS),
                    ColSpec(VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS),
                    ColSpec(VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE_FEATURES),
                    ColSpec(VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE,
                            VirchowMLFlowSchemaLiterals.OUTPUT_COLUMN_TEXT_FEATURES),
                ]
            )
        else:
            raise Exception("Unsupported task")

        return ModelSignature(inputs=input_schema, outputs=output_schema, params=params)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)

        from virchow_mlflow_model_wrapper import VirchowModelWrapper
        mlflow_model_wrapper = VirchowModelWrapper()

        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = self._get_code_path()

        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )

    def _get_code_path(self):
        """Return code path for saving mlflow model depending on task type.

        :return: code path
        :rtype: List[str]
        """
        code_path = [
            os.path.join(self.MODEL_DIR, "virchow_mlflow_model_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.COMMON_DIR, "vision_utils.py")
        ]

        return code_path

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dict for MLflow model.

        :return: artifacts dict
        :rtype: Dict
        """
        artifacts_dict = {
            VirchowMLflowLiterals.CHECKPOINT_PATH: self._model_dir+"/pytorch_model.bin",
            VirchowMLflowLiterals.CONFIG_PATH: self._model_dir+"/config.json"
        }
        return artifacts_dict


class HibouBMLFlowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLflow convertor for Hibou-B vision model."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "hibou_b")
    COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for Hibou-B model."""
        super().__init__(**kwargs)
        if self._task not in [SupportedTasks.IMAGE_FEATURE_EXTRACTION.value, SupportedTasks.FEATURE_EXTRACTION.value]:
            raise Exception("Unsupported task for Hibou-B convertor.")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the Hibou-B model."""
        input_schema = Schema([
            ColSpec(DataType.string, HibouBMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE)
        ])
        params = ParamSchema([
            ParamSpec(HibouBMLflowLiterals.DEVICE_TYPE, DataType.string, "cuda"),
            ParamSpec(HibouBMLflowLiterals.TO_HALF_PRECISION, DataType.boolean, False)
        ])
        output_schema = Schema([
            ColSpec(DataType.string, HibouBMLFlowSchemaLiterals.OUTPUT_COLUMN_IMAGE_FEATURES)
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema, params=params)

    def save_as_mlflow(self):
        """Prepare the Hibou-B model for saving to MLflow."""
        import sys

        sys.path.append(self.MODEL_DIR)
        from hiboub_mlflow_wrapper import HibouBPoolerMLFlowModelWrapper

        mlflow_model_wrapper = HibouBPoolerMLFlowModelWrapper(task_type=self._task)

        artifacts_dict = self._prepare_artifacts_dict()
        conda_env_file = os.path.join(self.MODEL_DIR, "conda.yaml")
        code_path = self._get_code_path()

        super()._save(
            mlflow_model_wrapper=mlflow_model_wrapper,
            artifacts_dict=artifacts_dict,
            conda_env=conda_env_file,
            code_path=code_path,
        )

    def _get_code_path(self):
        """Return a list of code file paths required to run the MLflow model."""
        code_path = [
            os.path.join(self.MODEL_DIR, "hiboub_mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
        ]
        return code_path

    def _prepare_artifacts_dict(self) -> Dict:
        """Prepare artifacts dictionary for the MLflow model.

        Assumes that the Hibou-B model directory contains:
          - pytorch_model.bin
          - config.json
        """
        import os
        print(os.listdir(self._model_dir))

        artifacts_dict = {
            HibouBMLflowLiterals.MODEL_DIR: self._model_dir
        }
        return artifacts_dict

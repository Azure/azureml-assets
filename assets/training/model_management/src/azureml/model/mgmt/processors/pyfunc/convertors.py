# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyFunc MLflow model convertors."""

import json
import mlflow
import os
import sys

from abc import ABC, abstractmethod
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow.types.schema import ColSpec, Schema
from pathlib import Path
from typing import Any, Dict, List

from azureml.model.mgmt.utils.logging_utils import get_logger
from azureml.model.mgmt.processors.convertors import MLFLowConvertorInterface
from azureml.model.mgmt.processors.pyfunc.config import MMLabDetectionTasks, SupportedTasks

from azureml.model.mgmt.processors.pyfunc.clip.config import \
    MLflowSchemaLiterals as CLIPMLFlowSchemaLiterals, MLflowLiterals as CLIPMLflowLiterals
from azureml.model.mgmt.processors.pyfunc.vision.config import \
    MLflowSchemaLiterals as VisionMLFlowSchemaLiterals, MMDetLiterals


logger = get_logger(__name__)


class PyFuncMLFLowConvertor(MLFLowConvertorInterface, ABC):
    """PyFunc MLflow convertor base class"""

    CONDA_FILE_NAME = "conda.yaml"
    REQUIREMENTS_FILE_NAME = "requirements.txt"

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
        self._model_dir = model_dir
        self._output_dir = output_dir
        self._temp_dir = temp_dir
        self._model_id = translate_params["model_id"]
        self._task = translate_params["task"]
        self._signatures = translate_params.get("signatures", None)

    def _save(
        self,
        mlflow_model_wrapper: PyFuncModel,
        artifacts_dict: Dict[str, str],
        pip_requirements: str,
        code_path: List[str],
    ):
        """Save Mlflow model to output directory.

        :param mlflow_model_wrapper: MLflow model wrapper instance
        :type mlflow_model_wrapper: Subclass of PyFuncModel
        :param artifacts_dict: Dictionary of name to artifact path
        :type artifacts_dict: Dict[str, str]
        :param pip_requirements: Path to pip requirements file
        :type pip_requirements: str
        :param code_path: A list of local filesystem paths to Python file dependencies
        :type code_path: List[str]

        """
        signatures = self._signatures or self.get_model_signature()

        mlflow.pyfunc.save_model(
            path=self._output_dir,
            python_model=mlflow_model_wrapper,
            artifacts=artifacts_dict,
            pip_requirements=pip_requirements,
            signature=signatures,
            code_path=code_path,
            metadata={"model_name": self._model_id},
        )

        logger.info("Model saved successfully.")

    def _validate(self, translate_params):
        """Validate translate parameters"""
        if not translate_params.get("task"):
            raise Exception("task is a required parameter for pyfunc flavor.")
        task = translate_params["task"]
        if not SupportedTasks.has_value(task):
            raise Exception(f"Unsupported task {task} for pyfunc flavor.")


class MMLabDetectionMLflowConvertor(PyFuncMLFLowConvertor):
    """PyFunc MLfLow convertor for detection models from MMLab."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "vision")

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
            [ColSpec(VisionMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE, VisionMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE)]
        )

        if self._task in [MMLabDetectionTasks.MM_OBJECT_DETECTION.value, MMLabDetectionTasks.MM_INSTANCE_SEGMENTATION.value]:
            output_schema = Schema(
                [
                    ColSpec(VisionMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE, VisionMLFlowSchemaLiterals.OUTPUT_COLUMN_BOXES),
                ]
            )
        else:
            raise NotImplementedError(f"Task type: {self._task} is not supported yet.")
        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from detection_predict import ImagesDetectionMLflowModelWrapper

        mlflow_model_wrapper = ImagesDetectionMLflowModelWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        pip_requirements = os.path.join(self.MODEL_DIR, "requirements.txt")
        code_path = [
            os.path.join(self.MODEL_DIR, "detection_predict.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
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

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for CLIP models."""
        super().__init__(**kwargs)
        if self._task != SupportedTasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value:
            raise Exception("Unsupported task")

    def get_model_signature(self) -> ModelSignature:
        """Return MLflow model signature with input and output schema for the given input task.

        :return: MLflow model signature.
        :rtype: mlflow.models.signature.ModelSignature
        """
        input_schema = Schema(
            [
                ColSpec(CLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE, CLIPMLFlowSchemaLiterals.INPUT_COLUMN_IMAGE),
                ColSpec(CLIPMLFlowSchemaLiterals.INPUT_COLUMN_TEXT_DATA_TYPE, CLIPMLFlowSchemaLiterals.INPUT_COLUMN_TEXT),
            ]
        )
        output_schema = Schema(
            [
                ColSpec(CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE, CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS),
                ColSpec(CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE, CLIPMLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS),
            ]
        )

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def save_as_mlflow(self):
        """Prepare model for save to MLflow."""
        sys.path.append(self.MODEL_DIR)
        from mlflow_wrapper import CLIPMLFlowModelWrapper

        mlflow_model_wrapper = CLIPMLFlowModelWrapper(task_type=self._task)
        artifacts_dict = self._prepare_artifacts_dict()
        pip_requirements = os.path.join(self.MODEL_DIR, "requirements.txt")
        code_path = [
            os.path.join(self.MODEL_DIR, "mlflow_wrapper.py"),
            os.path.join(self.MODEL_DIR, "config.py"),
            os.path.join(self.MODEL_DIR, "utils.py")
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
        artifacts_dict = {
            CLIPMLflowLiterals.MODEL_DIR: os.fspath(self._model_dir)
        }
        return artifacts_dict

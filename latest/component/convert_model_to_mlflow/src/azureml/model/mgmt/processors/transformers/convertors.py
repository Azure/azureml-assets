# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers MLflow model convertors."""

import transformers
import os
import yaml
from abc import ABC, abstractmethod
from azureml.evaluate import mlflow as hf_mlflow
from azureml.model.mgmt.processors.convertors import MLFLowConvertorInterface
from azureml.model.mgmt.processors.transformers.config import (
    HF_CONF,
    META_FILE_PATTERN,
    SupportedNLPTasks,
    SupportedTasks,
    SupportedVisionTasks,
)
from azureml.model.mgmt.utils.common_utils import (
    KV_EQ_SEP,
    ITEM_COMMA_SEP,
    copy_files,
    move_files,
    get_dict_from_comma_separated_str,
    get_list_from_comma_separated_str,
    run_command,
    fetch_mlflow_acft_metadata
)
from azureml.model.mgmt.utils.logging_utils import get_logger
from mlflow.models import ModelSignature, Model
from mlflow.types.schema import ColSpec
from mlflow.types.schema import DataType, Schema
from mlflow.utils.requirements_utils import _get_pinned_requirement
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoModelForMaskedLM,
    AutoModelForImageClassification,
    AutoTokenizer,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from typing import Any, Dict


logger = get_logger(__name__)


class HFMLFLowConvertor(MLFLowConvertorInterface, ABC):
    """HF MlfLow convertor base class."""

    CONDA_FILE_NAME = "conda.yaml"
    REQUIREMENTS_FILE_NAME = "requirements.txt"
    PREDICT_FILE_NAME = "predict.py"
    PREDICT_MODULE = "predict"

    @abstractmethod
    def get_model_signature(self):
        """Return model signature for MLflow model."""
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
        """Initialize MLflow convertor for HF models."""
        self._validate(translate_params)
        self._model_dir = model_dir
        self._output_dir = output_dir
        self._temp_dir = temp_dir
        self._model_id = translate_params.get("model_id", None)
        self._task = translate_params["task"]
        self._experimental = translate_params.get(HF_CONF.HF_USE_EXPERIMENTAL_FEATURES.value, False)
        self._misc = translate_params.get("misc", [])
        self._signatures = translate_params.get("signature", None)
        self._config_cls_name = translate_params.get(HF_CONF.HF_CONFIG_CLASS.value, None)
        self._model_cls_name = translate_params.get(HF_CONF.HF_PRETRAINED_CLASS.value, None)
        self._tokenizer_cls_name = translate_params.get(HF_CONF.HF_TOKENIZER_CLASS.value, None)
        self._extra_pip_requirements = get_list_from_comma_separated_str(
            translate_params.get(HF_CONF.EXTRA_PIP_REQUIREMENTS.value), ITEM_COMMA_SEP
        )

        if self._signatures:
            self._signatures = ModelSignature.from_dict(self._signatures)

        config_hf_load_kwargs = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_CONFIG_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP
        )
        tokenizer_hf_load_kwargs = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_TOKENIZER_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP
        )
        model_hf_load_args = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_MODEL_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP
        )
        pipeline_init_args = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_PIPELINE_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP
        )

        if pipeline_init_args and (model_hf_load_args or config_hf_load_kwargs or tokenizer_hf_load_kwargs):
            raise Exception("set(model, config, tokenizer) init args and pipeline init args are exclusive.")

        self._hf_conf = {HF_CONF.TASK_TYPE.value: self._task}
        if self._model_id:
            self._hf_conf[HF_CONF.HUGGINGFACE_ID.value] = self._model_id
        if pipeline_init_args:
            self._hf_conf[HF_CONF.HF_PIPELINE_ARGS.value] = pipeline_init_args
        if config_hf_load_kwargs:
            self._hf_conf[HF_CONF.HF_CONFIG_ARGS.value] = config_hf_load_kwargs
        if tokenizer_hf_load_kwargs:
            self._hf_conf[HF_CONF.HF_TOKENIZER_ARGS.value] = tokenizer_hf_load_kwargs
        if model_hf_load_args:
            self._hf_conf[HF_CONF.HF_MODEL_ARGS.value] = model_hf_load_args

        # class names need to be present in hf_config.
        # This gets updated in concrete subclasses.
        self._hf_config_cls = (
            None
            if not self._config_cls_name
            else self._get_transformers_class_from_str(class_name=self._config_cls_name)
        )
        self._hf_model_cls = (
            None
            if not self._model_cls_name
            else self._get_transformers_class_from_str(class_name=self._model_cls_name)
        )
        self._hf_tokenizer_cls = (
            None
            if not self._tokenizer_cls_name
            else self._get_transformers_class_from_str(class_name=self._tokenizer_cls_name)
        )

    def _get_transformers_class_from_str(self, class_name) -> Any:
        try:
            return getattr(transformers, class_name)
        except Exception as e:
            logger.error(f"Error in loading class {e}")
            return None

    def _save(
        self,
        conda_env=None,
        code_paths=None,
        input_example=None,
        requirements_file=None,
        pip_requirements=None,
        segregate=False,
    ):
        config = tokenizer = None
        model = str(self._model_dir)

        # try installing extra pip requirements
        if self._extra_pip_requirements or pip_requirements:
            pkgs = " ".join(self._extra_pip_requirements) if not pip_requirements else " ".join(pip_requirements)
            cmd = f"pip install {pkgs}"

            exit_code, stdout = run_command(cmd)
            if exit_code != 0:
                logger.warning(f"{pkgs} failed to install. Error:\n{stdout}\n")
            else:
                logger.info(f"Successully installed {pkgs}. pip logs =>\n{stdout}\n")

        if segregate:
            logger.info("Segregate input model dir and present into separate folders for model, config and tokenizer")
            logger.info("Preparing model files")
            tmp_model_dir = Path(self._temp_dir) / HF_CONF.HF_MODEL_PATH.value
            copy_files(self._model_dir, tmp_model_dir)
            model = str(tmp_model_dir)
            logger.info("Loading config")
            config = self._hf_config_cls.from_pretrained(
                self._model_dir, **self._hf_conf.get(HF_CONF.HF_CONFIG_ARGS.value, {})
            )
            logger.info("Loading tokenizer")
            tokenizer = self._hf_tokenizer_cls.from_pretrained(
                self._model_dir, **self._hf_conf.get(HF_CONF.HF_TOKENIZER_ARGS.value, {})
            )

        # Set experimental flag
        if self._experimental:
            logger.info("Experimental features enabled for MLflow conversion")
            self._hf_conf["exp"] = True

        # set metadata info
        metadata = fetch_mlflow_acft_metadata(base_model_name=self._model_id,
                                              is_finetuned_model=False)
        mlflow_model = Model(metadata=metadata)
        hf_mlflow.hftransformers.save_model(
            config=config,
            tokenizer=tokenizer,
            hf_model=model,
            hf_conf=self._hf_conf,
            mlflow_model=mlflow_model,
            conda_env=conda_env,
            code_paths=code_paths,
            signature=self._signatures,
            input_example=input_example,
            requirements_file=requirements_file,
            pip_requirements=pip_requirements,
            extra_pip_requirements=self._extra_pip_requirements,
            path=self._output_dir,
        )

        # move metadata files to parent folder
        logger.info("Moving meta files such as license, use_policy, readme to parent")
        move_files(
            Path(self._output_dir) / "data/model",
            self._output_dir,
            include_pattern_str=META_FILE_PATTERN,
            ignore_case=True
        )

        # pin pycocotools==2.0.4
        self._update_conda_dependencies({"pycocotools": "2.0.4"})

    def _update_conda_dependencies(self, package_details):
        """Update conda dependencies.

        Sample:
            pkg_details = {"pycocotools": "2.0.4"}
        """
        conda_file_path = os.path.join(self._output_dir, HFMLFLowConvertor.CONDA_FILE_NAME)

        if not os.path.exists(conda_file_path):
            logger.warning("Malformed mlflow model. Please make sure conda.yaml exists")
            return

        with open(conda_file_path) as f:
            conda_dict = yaml.safe_load(f)

        conda_deps = conda_dict["dependencies"]
        for i in range(len(conda_deps)):
            if isinstance(conda_deps[i], str):
                pkg_name = conda_deps[i].split("=")[0]
                if pkg_name in package_details:
                    pkg_version = package_details[pkg_name]
                    logger.info(f"updating with {pkg_name}={pkg_version}")
                    conda_deps[i] = f"{pkg_name}={pkg_version}"
                    package_details.pop(pkg_name)

        for pkg_name, pkg_version in package_details.items():
            logger.info(f"adding  {pkg_name}={pkg_version}")
            conda_deps.append(f"{pkg_name}={pkg_version}")

        with open(conda_file_path, "w") as f:
            yaml.safe_dump(conda_dict, f)
            logger.info("updated conda.yaml")

    def _validate(self, translate_params):
        if not translate_params.get("task"):
            raise Exception("task is a required parameter for hftransformers MLflow flavor.")
        task = translate_params["task"]
        if not SupportedTasks.has_value(task):
            raise Exception(f"Unsupported task {task} for hftransformers MLflow flavor.")


class VisionMLflowConvertor(HFMLFLowConvertor):
    """HF MlfLow convertor for vision models."""

    VISION_DIR = Path(__file__).parent / "vision"
    PREDICT_FILE_PATH = VISION_DIR / HFMLFLowConvertor.PREDICT_FILE_NAME
    VISION_UTILS_FILE_PATH = Path(__file__).parent.parent / "common" / "vision_utils.py"

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for vision models."""
        super().__init__(**kwargs)
        if not SupportedVisionTasks.has_value(self._task):
            raise Exception("Unsupported vision task")

    def get_model_signature(self):
        """Return model signature for vision models."""
        return ModelSignature(
            inputs=Schema(inputs=[ColSpec(name="image", type=DataType.binary)]),
            outputs=Schema(
                inputs=[ColSpec(name="probs", type=DataType.string), ColSpec(name="labels", type=DataType.string)]
            ),
        )

    def save_as_mlflow(self):
        """Prepare vision models for save to MLflow."""
        hf_conf = self._hf_conf
        self._hf_model_cls = self._hf_model_cls if self._hf_model_cls else AutoModelForImageClassification
        self._hf_config_cls = self._hf_config_cls if self._hf_config_cls else AutoConfig
        self._hf_tokenizer_cls = self._hf_tokenizer_cls if self._hf_tokenizer_cls else AutoImageProcessor
        self._signatures = self._signatures or self.get_model_signature()

        hf_conf[HF_CONF.HF_CONFIG_CLASS.value] = self._hf_config_cls.__name__
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_TOKENIZER_CLASS.value] = self._hf_tokenizer_cls.__name__
        hf_conf[HF_CONF.HF_PREDICT_MODULE.value] = HFMLFLowConvertor.PREDICT_MODULE

        config_load_args = self._hf_conf.get(HF_CONF.HF_CONFIG_ARGS.value, {})
        config = self._hf_config_cls.from_pretrained(self._model_dir, local_files_only=True, **config_load_args)
        hf_conf[HF_CONF.TRAIN_LABEL_LIST.value] = list(config.id2label.values())
        extra_pip_requirements = ["torchvision"]
        if self._extra_pip_requirements is None:
            self._extra_pip_requirements = []
        for package_name in extra_pip_requirements:
            package_with_version = _get_pinned_requirement(package_name)
            self._extra_pip_requirements.append(package_with_version)

        return super()._save(
            code_paths=[VisionMLflowConvertor.PREDICT_FILE_PATH, VisionMLflowConvertor.VISION_UTILS_FILE_PATH],
            segregate=True,
        )


class ASRMLflowConvertor(HFMLFLowConvertor):
    """HF MlfLow convertor base class for ASR models."""

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for ASR models."""
        super().__init__(**kwargs)
        if self._task != SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value:
            raise Exception(f"Unsupported ASR task {self._task}")

    def get_model_signature(self):
        """Return model signature for ASR models."""
        return ModelSignature(
            inputs=Schema(
                inputs=[ColSpec(name="audio", type=DataType.string), ColSpec(name="language", type=DataType.string)]
            ),
            outputs=Schema(inputs=[ColSpec(name="text", type=DataType.string)]),
        )


class WhisperMLflowConvertor(ASRMLflowConvertor):
    """HF MlfLow convertor base class for ASR models."""

    MODEL_FAMILY = "whisper"
    WHISPER_DIR = Path(__file__).parent / "whisper"
    PREDICT_FILE_PATH = WHISPER_DIR / HFMLFLowConvertor.PREDICT_FILE_NAME
    CONDA_FILE_PATH = WHISPER_DIR / HFMLFLowConvertor.CONDA_FILE_NAME

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for whisper model."""
        super().__init__(**kwargs)

    def save_as_mlflow(self):
        """Prepare Whisper model for save to MLflow."""
        hf_conf = self._hf_conf
        self._hf_model_cls = self._hf_model_cls if self._hf_model_cls else WhisperForConditionalGeneration
        self._hf_config_cls = self._hf_config_cls if self._hf_config_cls else WhisperConfig
        self._hf_tokenizer_cls = self._hf_tokenizer_cls if self._hf_tokenizer_cls else WhisperProcessor

        hf_conf[HF_CONF.HF_CONFIG_CLASS.value] = self._hf_config_cls.__name__
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_TOKENIZER_CLASS.value] = self._hf_tokenizer_cls.__name__
        hf_conf[HF_CONF.HF_PREDICT_MODULE.value] = HFMLFLowConvertor.PREDICT_MODULE

        conda_env = {}
        with open(WhisperMLflowConvertor.CONDA_FILE_PATH) as f:
            conda_env = yaml.safe_load(f)

        return super()._save(
            conda_env=conda_env,
            code_paths=[WhisperMLflowConvertor.PREDICT_FILE_PATH],
            segregate=True,
        )


class NLPMLflowConvertor(HFMLFLowConvertor):
    """HF MLflow convertor for NLP models."""

    TASK_TO_HF_PRETRAINED_CLASS_MAPPING = {
        "fill-mask": AutoModelForMaskedLM,
        "text-classification": AutoModelForSequenceClassification,
        "token-classification": AutoModelForTokenClassification,
        "question-answering": AutoModelForQuestionAnswering,
        "summarization": AutoModelWithLMHead,
        "text-generation": AutoModelWithLMHead,
        "translation": AutoModelForSeq2SeqLM,
    }

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for NLP models."""
        super().__init__(**kwargs)
        if not SupportedNLPTasks.has_value(self._task):
            raise Exception("Unsupported NLP task")

    def get_model_signature(self):
        """Return model signature for NLP models."""
        if self._task == SupportedNLPTasks.QUESTION_ANSWERING.value:
            return ModelSignature(
                inputs=Schema(
                    inputs=[
                        ColSpec(name="question", type=DataType.string),
                        ColSpec(name="context", type=DataType.string),
                    ]
                ),
                outputs=Schema(inputs=[ColSpec(name="text", type=DataType.string)]),
            )

        return ModelSignature(
            inputs=Schema(inputs=[ColSpec(name="input_string", type=DataType.string)]),
            outputs=Schema(inputs=[ColSpec(name="text", type=DataType.string)]),
        )

    def save_as_mlflow(self):
        """Prepate NLP model for save to MLflow."""
        hf_conf = self._hf_conf
        self._hf_model_cls = (
            self._hf_model_cls
            if self._hf_model_cls
            else NLPMLflowConvertor.TASK_TO_HF_PRETRAINED_CLASS_MAPPING.get(self._task) or AutoModel
        )
        self._hf_config_cls = self._hf_config_cls if self._hf_config_cls else AutoConfig
        self._hf_tokenizer_cls = self._hf_tokenizer_cls if self._hf_tokenizer_cls else AutoTokenizer

        hf_conf[HF_CONF.HF_CONFIG_CLASS.value] = self._hf_config_cls.__name__
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_TOKENIZER_CLASS.value] = self._hf_tokenizer_cls.__name__

        return super()._save(segregate=True)

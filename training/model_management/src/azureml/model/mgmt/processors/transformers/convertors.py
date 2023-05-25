# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers mlflow model convertors."""

import transformers
import yaml
from abc import ABC, abstractmethod
from azureml.evaluate import mlflow as hf_mlflow
from azureml.model.mgmt.processors.transformers.config import (
    HF_CONF,
    MODEL_FILE_PATTERN,
    SupportedNLPTasks,
    SupportedTasks,
    SupportedVisionTasks,
)
from azureml.model.mgmt.utils.common_utils import (
    KV_EQ_SEP,
    ITEM_COMMA_SEP,
    copy_file_paths_to_destination,
    get_dict_from_comma_separated_str,
    get_list_from_comma_separated_str,
)
from diffusers import StableDiffusionPipeline
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec
from mlflow.types.schema import DataType, Schema
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


class HFMLFLowConvertor(ABC):
    """HF MlfLow convertor base class."""

    CONDA_FILE_NAME = "conda.yaml"
    PREDICT_FILE_NAME = "predict.py"
    PREDICT_MODULE = "predict"

    @abstractmethod
    def get_model_signature(self):
        """Return model signature for mlflow model."""
        raise NotImplementedError

    @abstractmethod
    def save_as_mlflow(self):
        """Prepare model for save to mlflow."""
        raise NotImplementedError

    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        translate_params: Dict,
    ):
        """Initialize mlflow convertor for HF models."""
        self._model_dir = model_dir
        self._output_dir = output_dir
        self._model_id = translate_params["model_id"]
        self._task = translate_params["task"]
        self._misc = translate_params["misc"]
        self._signatures = translate_params.get("signature", None)
        self._config_cls_name = translate_params.get(HF_CONF.HF_CONFIG_CLASS.value, None)
        self._model_cls_name = translate_params.get(HF_CONF.HF_PRETRAINED_CLASS.value, None)
        self._tokenizer_cls_name = translate_params.get(HF_CONF.HF_TOKENIZER_CLASS.value, None)
        self._extra_pip_requirements = get_list_from_comma_separated_str(
            translate_params.get(HF_CONF.EXTRA_PIP_DEPENDENCIES.value), ITEM_COMMA_SEP
        )

        if self._signatures:
            self._signatures = ModelSignature.from_dict(self._signatures)

        config_hf_load_kwargs = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_CONFIG_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP, do_eval=True
        )
        tokenizer_hf_load_kwargs = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_TOKENIZER_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP, do_eval=True
        )
        model_hf_load_args = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_MODEL_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP, do_eval=True
        )
        pipeline_init_args = get_dict_from_comma_separated_str(
            translate_params.get(HF_CONF.HF_PIPELINE_ARGS.value), ITEM_COMMA_SEP, KV_EQ_SEP, do_eval=True
        )

        if pipeline_init_args and (model_hf_load_args or config_hf_load_kwargs or tokenizer_hf_load_kwargs):
            raise Exception("set(model, config, tokenizer) init args and pipeline init args are exclusive.")

        self._hf_conf = {
            HF_CONF.TASK_TYPE.value: self._task,
            HF_CONF.HUGGINGFACE_ID.value: self._model_id,
        }

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
            print(f"Error in loading class {e}")
            return None

    def _save(
        self,
        config=None,
        tokenizer=None,
        conda_env=None,
        code_paths=None,
        input_example=None,
        requirements_file=None,
        pip_requirements=None,
        extra_pip_requirements=None,
    ):
        if config or tokenizer:
            print("Either of config and tokenizer is not null. Filtering out model files to save.")
            tmp_model_dir = Path("tmp") / "model"
            copy_file_paths_to_destination(self._model_dir, tmp_model_dir, MODEL_FILE_PATTERN)
            self._model_dir = tmp_model_dir

        self._signatures = self._signatures or self.get_model_signature()
        hf_mlflow.hftransformers.save_model(
            config=config,
            tokenizer=tokenizer,
            hf_model=str(self._model_dir),
            hf_conf=self._hf_conf,
            conda_env=conda_env,
            code_paths=code_paths,
            signature=self._signatures,
            input_example=input_example,
            requirements_file=requirements_file,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            path=self._output_dir,
        )


class VisionMLflowConvertor(HFMLFLowConvertor):
    """HF MlfLow convertor for vision models."""

    VISION_DIR = Path().parent / "vision"
    PREDICT_FILE_PATH = VISION_DIR / HFMLFLowConvertor.PREDICT_FILE_NAME
    CONDA_FILE_PATH = VISION_DIR / HFMLFLowConvertor.CONDA_FILE_NAME

    def __init__(self, **kwargs):
        """Initialize mlflow convertor for vision models."""
        super().__init__(**kwargs)
        if not SupportedVisionTasks.has_value(self._task):
            raise Exception("Unsupported vision task")

    def get_model_signature(self):
        """Return model signature for vision models."""
        return ModelSignature(
            inputs=Schema(inputs=[ColSpec(name="image", type=DataType.string)]),
            outputs=Schema(
                inputs=[ColSpec(name="probs", type=DataType.string), ColSpec(name="labels", type=DataType.string)]
            ),
        )

    def save_as_mlflow(self):
        """Prepare vision models for save to mlflow."""
        hf_conf = self._hf_conf
        self._hf_model_cls = self._hf_model_cls if self._hf_model_cls else AutoModelForImageClassification
        self._hf_config_cls = self._hf_config_cls if self._hf_config_cls else AutoConfig
        self._hf_tokenizer_cls = self._hf_tokenizer_cls if self._hf_tokenizer_cls else AutoImageProcessor

        hf_conf[HF_CONF.HF_CONFIG_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_TOKENIZER_CLASS.value] = self._hf_tokenizer_cls.__name__
        hf_conf[HF_CONF.HF_PREDICT_MODULE.value] = HFMLFLowConvertor.PREDICT_MODULE

        config_load_args = self._hf_conf[HF_CONF.HF_CONFIG_ARGS.value]
        tokenizer_load_args = self._hf_conf[HF_CONF.HF_TOKENIZER_ARGS.value]
        config = self._hf_config_cls.from_pretrained(self._model_dir, local_files_only=True, **config_load_args)
        tokenizer = self._hf_config_cls.from_pretrained(
            self._model_dir, config=config, local_files_only=True, **tokenizer_load_args
        )

        hf_conf[HF_CONF.TRAIN_LABEL_LIST.value] = list(config.id2label.values())

        return super()._save(
            config=config,
            tokenizer=tokenizer,
            code_paths=[VisionMLflowConvertor.PREDICT_FILE_PATH],
        )


class ASRMLflowConvertor(HFMLFLowConvertor):
    """HF MlfLow convertor base class for ASR models."""

    def __init__(self, **kwargs):
        """Initialize mlflow convertor for ASR models."""
        super().__init__(**kwargs)
        if self.task != SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value:
            raise Exception(f"Unsupported ASR task {self.task}")

    def get_model_signature(self):
        """Return model signature for ASR models."""
        return ModelSignature(
            inputs=Schema(
                inputs=[ColSpec(name="audio", type=DataType.string), ColSpec(name="language", type=DataType.string)]
            ),
            outputs=Schema(inputs=[ColSpec(name="text", type=DataType.string)]),
        )


class WhisperMLFlowConvertor(ASRMLflowConvertor):
    """HF MlfLow convertor base class for ASR models."""

    MODEL_FAMILY = "whisper"
    WHISPER_DIR = Path().parent / "whisper"
    PREDICT_FILE_PATH = WHISPER_DIR / HFMLFLowConvertor.PREDICT_FILE_NAME
    CONDA_FILE_PATH = WHISPER_DIR / HFMLFLowConvertor.CONDA_FILE_NAME

    def __init__(self, **kwargs):
        """Initialize mlflow convertor for whisper model."""
        super().__init__(**kwargs)

    def save_as_mlflow(self):
        """Prepare Whisper model for save to mlflow."""
        hf_conf = self._hf_conf
        self._hf_model_cls = self._hf_model_cls if self._hf_model_cls else WhisperForConditionalGeneration
        self._hf_config_cls = self._hf_config_cls if self._hf_config_cls else WhisperConfig
        self._hf_tokenizer_cls = self._hf_tokenizer_cls if self._hf_tokenizer_cls else WhisperProcessor

        hf_conf[HF_CONF.HF_CONFIG_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_TOKENIZER_CLASS.value] = self._hf_tokenizer_cls.__name__
        hf_conf[HF_CONF.HF_PREDICT_MODULE.value] = HFMLFLowConvertor.PREDICT_MODULE

        hf_conf[HF_CONF.HF_TOKENIZER_ARGS].update({"padding": True, "truncation": True})

        config_load_args = self._hf_conf[HF_CONF.HF_CONFIG_ARGS.value]
        tokenizer_load_args = self._hf_conf[HF_CONF.HF_TOKENIZER_ARGS.value]
        config = self._hf_config_cls.from_pretrained(self._model_dir, local_files_only=True, **config_load_args)
        tokenizer = self._hf_config_cls.from_pretrained(
            self._model_dir, padding=True, truncation=True, local_files_only=True, **tokenizer_load_args
        )

        conda_env = yaml.safe_load(WhisperMLFlowConvertor.CONDA_FILE_PATH)
        return super()._save(
            config=config,
            tokenizer=tokenizer,
            conda_env=conda_env,
            code_paths=[WhisperMLFlowConvertor.PREDICT_FILE_PATH],
        )


class TextToImageDiffuserMLFlowConvertor(HFMLFLowConvertor):
    """HF MlfLow convertor base class for text to image diffuser models."""

    def __init__(self, **kwargs):
        """Initialize mlflow convertor for t2image models."""
        super().__init__(**kwargs)

    def get_model_signature(self):
        """Return model signature for text to image models."""
        return ModelSignature(
            inputs=Schema(inputs=[ColSpec(name="input_string", type=DataType.string)]),
            outputs=Schema(inputs=[ColSpec(name="image", type=DataType.string)]),
        )


class StableDiffusionMlflowConvertor(TextToImageDiffuserMLFlowConvertor):
    """HF MlfLow convertor class for stable diffusion models."""

    def __init__(self, **kwargs):
        """Initialize mlflow convertor for SD models."""
        super().__init__(**kwargs)

    def save_as_mlflow(self):
        """Prepare SD model for save to mlflow."""
        hf_conf = self._hf_conf
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = StableDiffusionPipeline
        hf_conf[HF_CONF.CUSTOM_CONFIG_MODULE] = "diffusers"
        hf_conf[HF_CONF.CUSTOM_MODLE_MODULE] = "diffusers"
        hf_conf[HF_CONF.CUSTOM_TOKENIZER_MODULE] = "diffusers"
        hf_conf[HF_CONF.FORCE_LOAD_CONFIG] = False
        hf_conf[HF_CONF.FORCE_LOAD_TOKENIZER] = False
        return super()._save()


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
        """Initialize mlflow convertor for NLP models."""
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
        """Prepate NLP model for save to mlflow."""
        hf_conf = self._hf_conf
        self._hf_model_cls = (
            self._hf_model_cls
            if self._hf_model_cls
            else NLPMLflowConvertor.TASK_TO_HF_PRETRAINED_CLASS_MAPPING.get(self._task) or AutoModel
        )
        self._hf_config_cls = self._hf_config_cls if self._hf_config_cls else AutoConfig
        self._hf_tokenizer_cls = self._hf_tokenizer_cls if self._hf_tokenizer_cls else AutoTokenizer

        hf_conf[HF_CONF.HF_CONFIG_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_TOKENIZER_CLASS.value] = self._hf_tokenizer_cls.__name__

        config_load_args = self._hf_conf[HF_CONF.HF_CONFIG_ARGS.value]
        tokenizer_load_args = self._hf_conf[HF_CONF.HF_TOKENIZER_ARGS.value]
        config = self._hf_config_cls.from_pretrained(self._model_dir, local_files_only=True, **config_load_args)
        tokenizer = self._hf_config_cls.from_pretrained(self._model_dir, local_files_only=True, **tokenizer_load_args)

        return super()._save(config=config, tokenizer=tokenizer)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers MLflow model convertors."""
from typing import List
import base64
import io
import transformers
import platform
import mlflow
import numpy as np
import os
import yaml
from abc import ABC, abstractmethod
from PIL import Image
from azureml.evaluate import mlflow as hf_mlflow
from azureml.core.conda_dependencies import CondaDependencies
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
    get_mlclient,
    get_dict_from_comma_separated_str,
    get_list_from_comma_separated_str,
    fetch_mlflow_acft_metadata
)
from azureml.model.mgmt.utils.logging_utils import get_logger
from mlflow.models import ModelSignature, Model, infer_signature
from mlflow.types.schema import ColSpec, DataType, Schema, ParamSpec, ParamSchema
from mlflow.transformers import generate_signature_output, get_default_conda_env
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
    AutoModelForCausalLM,
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
        self._task = translate_params.get("task", None)
        self._model_flavor = translate_params.get("model_flavor", "HFTransformersV2")
        self._vllm_enabled = translate_params.get("vllm_enabled", False)
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
        model = str(self._model_dir)

        # set metadata info
        metadata = fetch_mlflow_acft_metadata(base_model_name=self._model_id,
                                              is_finetuned_model=False,
                                              base_model_task=self._task)

        mlclient = get_mlclient("azureml")
        if self._vllm_enabled:
            vllm_image = mlclient.environments.get("foundation-model-inference", label="latest")
            metadata["azureml.base_image"] = "mcr.microsoft.com/azureml/curated/foundation-model-inference:" \
                + str(vllm_image.version)

        logger.info("Metadata: {}".format(metadata))

        if self._model_flavor == "OSS":
            try:
                self._save_in_oss_flavor(model, metadata, conda_env, code_paths, input_example, pip_requirements)
            except Exception as e:
                logger.error("Model save failed with mlflow OSS flow for task: {} "
                             "with exception: {}".format(self._task, e))

                self._save_in_hftransformersv2_flavor(metadata, conda_env,
                                                      code_paths, input_example,
                                                      requirements_file, pip_requirements)
        else:
            self._save_in_hftransformersv2_flavor(metadata, conda_env,
                                                  code_paths, input_example,
                                                  requirements_file, pip_requirements)

        # pin pycocotools==2.0.4
        self._update_conda_dependencies({"pycocotools": "2.0.4"})

    def _save_in_oss_flavor(self, model, metadata, conda_env, code_paths, input_example, pip_requirements):
        # create a conda environment for OSS transformers Flavor

        curated_conda_env = conda_env
        if not self._extra_pip_requirements and not pip_requirements:
            python_version = platform.python_version()
            pip_pkgs = self._get_curated_environment_pip_package_list()
            conda_deps = CondaDependencies.create(conda_packages=None,
                                                  python_version=python_version,
                                                  pip_packages=pip_pkgs,
                                                  pin_sdk_version=False)

            curated_conda_env = conda_env or conda_deps.as_dict()
            pip_requirements = None

        # handle OSS trust_remote_code value
        trust_remote_code_val = False
        config_args = self._hf_conf.get(HF_CONF.HF_CONFIG_ARGS.value, {})
        tokenizer_args = self._hf_conf.get(HF_CONF.HF_TOKENIZER_ARGS.value, {})
        model_args = self._hf_conf.get(HF_CONF.HF_MODEL_ARGS.value, {})

        if (config_args.get('trust_remote_code', False)
                and tokenizer_args.get('trust_remote_code', False)
                and model_args.get('trust_remote_code', False)):
            trust_remote_code_val = True

        model_pipeline = transformers.pipeline(task=self._task, model=model,
                                               trust_remote_code=trust_remote_code_val)
        if hasattr(self, "config"):
            # Have to update the config as some vision models have problems in hf registry.
            model_pipeline.model.config = self.config
        params = {
            "transformers_model": model_pipeline,
            "code_paths": code_paths,
            "signature": self._signatures,
            "input_example": input_example,
            "metadata": metadata,
            "path": str(self._output_dir),
        }

        if curated_conda_env:
            params.update({
                "conda_env": curated_conda_env,
            })
        elif pip_requirements:
            params.update({
                "pip_requirements": pip_requirements,
            })
        else:
            params.update({
                "extra_pip_requirements": self._extra_pip_requirements,
            })

        mlflow.transformers.save_model(**params)

        logger.info("Model saved with mlflow OSS flow for task: {}".format(self._task))

    def _save_in_hftransformersv2_flavor(self, metadata, conda_env,
                                         code_paths, input_example,
                                         requirements_file, pip_requirements):

        # Set experimental flag
        if self._experimental:
            logger.info("Experimental features enabled for MLflow conversion")
            self._hf_conf["exp"] = True

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

        logger.info("Model saved with transformers evaluate flow for task: {}".format(self._task))

        # move metadata files to parent folder
        logger.info("Moving meta files such as license, use_policy, readme to parent")
        move_files(
            Path(self._output_dir) / "data/model",
            self._output_dir,
            include_pattern_str=META_FILE_PATTERN,
            ignore_case=True
        )

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

    def _get_curated_environment_pip_package_list(self) -> List[str]:
        """
        Retrieve the packages using 'conda list' command.

        :return: A List of the pip package and the corresponding versions.
        """
        import subprocess
        import json

        PIP_LIST = ['accelerate', 'cffi', 'dill', 'google-api-core', 'numpy',
                    'packaging', 'pillow', 'protobuf', 'pyyaml', 'requests', 'scikit-learn',
                    'scipy', 'sentencepiece', 'torch', 'mlflow']
        ADD_PACKAGE_LIST = ['torchvision==0.14.1', 'transformers==4.35.2']

        conda_list_cmd = ["conda", "list", "--json"]
        try:
            process = subprocess.run(conda_list_cmd, shell=False, check=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (FileNotFoundError, subprocess.CalledProcessError) as err:
            logger.warning('subprocess failed to get dependencies list from conda with error: {}'.format(err))
            return []
        output_str = process.stdout.decode('ascii')
        output_json = json.loads(output_str)
        pip_list = []
        for pkg in output_json:
            pkg_name = pkg['name']
            pkg_version = pkg['version']
            if pkg_name in PIP_LIST:
                pip_list.append(pkg_name + "==" + pkg_version)

        pip_list.extend(ADD_PACKAGE_LIST)

        logger.info("pip list: {}".format(pip_list))
        return pip_list

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

    def _robust_load_config(self, config) -> transformers.PretrainedConfig:
        """Check and modify if config has missing indices.

        :param config: Config Object from transformers
        :type config: PretrainedConfig
        :return: Config Object
        :rtype: PretrainedConfig
        """
        ids = list(config.id2label.keys())
        ids.sort()
        if max(ids) != len(ids)-1:
            missing_keys = set([x for x in range(max(ids))]).difference(set(ids))
            id2label = {}
            for idx, id in enumerate(ids):
                id2label[idx] = config.id2label[id]
            config.id2label = id2label
            self.config = config
            logger.warning(f"config loaded with modified id2label as there are some missing keys : {missing_keys}.")
        return config

    def get_random_base64_decoded_image(self) -> str:
        """Get random base64 decoded image.

        :return: base64 decoded image
        :rtype: string
        """
        imarray = np.random.rand(100, 100, 3) * 255
        buffered = io.BytesIO()
        image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

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
        config = self._robust_load_config(config)
        hf_conf[HF_CONF.TRAIN_LABEL_LIST.value] = list(config.id2label.values())
        extra_pip_requirements = ["torchvision"]
        if self._extra_pip_requirements is None:
            self._extra_pip_requirements = []
        for package_name in extra_pip_requirements:
            package_with_version = _get_pinned_requirement(package_name)
            self._extra_pip_requirements.append(package_with_version)

        if self._model_flavor == "OSS":
            vision_model = transformers.pipeline(task=self._task, model=str(self._model_dir))
            vision_model.model.config = config
            image_str = self.get_random_base64_decoded_image()
            self._signatures = infer_signature(
                image_str, generate_signature_output(vision_model, image_str),
            )
            return super()._save(
                conda_env=get_default_conda_env(vision_model.model)
            )
        else:
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
        "summarization": AutoModelForSeq2SeqLM,
        "chat-completion": AutoModelForCausalLM,
        "text-generation": AutoModelForCausalLM,
        "translation": AutoModelForSeq2SeqLM,
    }

    def __init__(self, **kwargs):
        """Initialize MLflow convertor for NLP models."""
        super().__init__(**kwargs)
        if not SupportedNLPTasks.has_value(self._task):
            raise Exception("Unsupported NLP task")

    def get_model_signature(self):
        """Return model signature for NLP models."""
        if self._task == SupportedNLPTasks.TEXT_GENERATION.value or self._task == SupportedNLPTasks.CHAT_COMPLETION:
            inputs = Schema([ColSpec(DataType.string)])
            outputs = Schema([ColSpec(DataType.string)])
            params = ParamSchema([ParamSpec("top_p", "float", default=0.8),
                                  ParamSpec("temperature", "float", default=0.8),
                                  ParamSpec("max_new_tokens", "integer", default=100),
                                  ParamSpec("do_sample", "boolean", default=True),
                                  ParamSpec("return_full_text", "boolean", default=True)])
            return ModelSignature(inputs=inputs, outputs=outputs, params=params)

        return self._signatures

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
        if self._model_flavor == "OSS":
            self._signatures = self._signatures or self.get_model_signature()

        hf_conf[HF_CONF.HF_CONFIG_CLASS.value] = self._hf_config_cls.__name__
        hf_conf[HF_CONF.HF_PRETRAINED_CLASS.value] = self._hf_model_cls.__name__
        hf_conf[HF_CONF.HF_TOKENIZER_CLASS.value] = self._hf_tokenizer_cls.__name__

        return super()._save(segregate=True)

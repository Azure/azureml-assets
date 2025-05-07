# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model converter adapters."""

from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
import json
import shutil
import yaml
import os

from azureml.acft.accelerator.utils.code_utils import (
    get_model_custom_code_files,
    copy_code_files,
)
from azureml.acft.accelerator.utils.license_utils import download_license_file
from azureml.acft.accelerator.utils.run_utils import is_main_process


from azureml.acft.common_components.utils.mlflow_utils import update_acft_metadata

from azureml.acft.contrib.hf.nlp.utils.common_utils import deep_update
from azureml.acft.contrib.hf.nlp.constants.constants import (
    MLFlowHFFlavourConstants,
    MLFlowHFFlavourTasks,
    SaveFileConstants,
    HfModelTypes,
    Tasks,
)

import mlflow
from mlflow.models import Model
import azureml.evaluate.mlflow as hf_mlflow

from transformers import pipeline

from azureml.acft.common_components import get_logger_app, ModelSelectorConstants

from model_converter_utils import (
    load_model,
    load_tokenizer,
    copy_tokenizer_files_to_model_folder,
)


logger = get_logger_app(
    "azureml.acft.contrib.hf.scripts.src.model_converter.model_converter_adapters"
)

PREPROCESSOR_CONFIG = "preprocessor_config.json"

MLFLOW_TASK_HF_PRETRAINED_CLASS_MAP = {
    MLFlowHFFlavourTasks.SINGLE_LABEL_CLASSIFICATION: "AutoModelForSequenceClassification",
    MLFlowHFFlavourTasks.MULTI_LABEL_CLASSIFICATION: "AutoModelForSequenceClassification",
    MLFlowHFFlavourTasks.NAMED_ENTITY_RECOGNITION: "AutoModelForTokenClassification",
    MLFlowHFFlavourTasks.QUESTION_ANSWERING: "AutoModelForQuestionAnswering",
    MLFlowHFFlavourTasks.TEXT_GENERATION: "AutoModelForCausalLM",
    MLFlowHFFlavourTasks.SUMMARIZATION: "AutoModelForSeq2SeqLM",
    MLFlowHFFlavourTasks.TRANSLATION: "AutoModelForSeq2SeqLM",
    MLFlowHFFlavourTasks.CHAT_COMPLETION: "AutoModelForCausalLM",
}
UNWANTED_PACKAGES = ["apex>", "apex<", "apex="]


class ModelConverter(ABC):
    """Base model converter class."""

    def __init__(self) -> None:
        """Init."""
        pass

    @abstractmethod
    def convert_model(self, *args, **kwargs) -> None:
        """Convert model format."""
        pass

    def download_license_file(
        self, model_name: str, src_model_path: str, dst_model_path: str
    ):
        """Save LICENSE file to MlFlow model."""
        license_file_path = Path(src_model_path, MLFlowHFFlavourConstants.LICENSE_FILE)
        # check if pytorch model has LICENSE file
        if license_file_path.is_file():
            shutil.copy(str(license_file_path), dst_model_path)
            logger.info("LICENSE file is copied to mlflow model folder.")
        else:
            # else download from hub if possible
            download_license_file(model_name, dst_model_path)

    def copy_finetune_config(self, src_model_path: str, dst_model_path: str):
        """Copy finetune_config.json to mlflow model artifacts."""
        finetune_config_path = Path(
            src_model_path, SaveFileConstants.ACFT_CONFIG_SAVE_PATH
        )
        if finetune_config_path.is_file():
            shutil.copyfile(
                str(finetune_config_path),
                str(Path(dst_model_path, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)),
            )
            logger.info(f"Copied {SaveFileConstants.ACFT_CONFIG_SAVE_PATH} file.")

    def copy_ml_configs(self, src_model_path: str, dst_model_path: str):
        """Copy ml_configs folder to mlflow model artifacts."""
        ml_configs_path = Path(src_model_path, "ml_configs")
        dst_model_path = Path(dst_model_path, "ml_configs")
        if ml_configs_path.is_dir():
            shutil.copytree(ml_configs_path, dst_model_path)
        logger.info("Copied ml_configs folder.")

    def copy_preprocessor_config(self, src_model_path: str, dst_model_path: str):
        """Copy preprocessor_config.json to mlflow model artifacts."""
        preprocessor_config_path = Path(src_model_path, PREPROCESSOR_CONFIG)
        if preprocessor_config_path.is_file():
            shutil.copyfile(
                str(preprocessor_config_path),
                str(Path(dst_model_path, "data", "model", PREPROCESSOR_CONFIG)),
            )
            logger.info(f"Copied {PREPROCESSOR_CONFIG} file.")
        else:
            logger.info(f"{PREPROCESSOR_CONFIG} file not found.")


class PyTorch_to_MlFlow_ModelConverter:
    """Mixin class to convert pytorch to hftransformers/oss flavour mlflow model."""

    def __init__(self, component_args: Namespace) -> None:
        """Init."""
        self.component_args = component_args
        self.ft_config = getattr(component_args, "ft_config", {})
        self.ft_pytorch_model_path = component_args.model_path
        self.mlflow_model_save_path = self.component_args.output_dir

    def _set_mlflow_hf_args(self):
        # set mlflow hf args
        self.mlflow_hf_args = {
            "hf_config_class": "AutoConfig",
            "hf_tokenizer_class": "AutoTokenizer",
        }
        for task in MLFLOW_TASK_HF_PRETRAINED_CLASS_MAP.keys():
            if self.mlflow_task_type == task or self.mlflow_task_type.startswith(task):
                self.mlflow_hf_args["hf_pretrained_class"] = (
                    MLFLOW_TASK_HF_PRETRAINED_CLASS_MAP[task]
                )
        # Override for llama-4 chat completion
        if (
            self.mlflow_task_type == MLFlowHFFlavourTasks.CHAT_COMPLETION
            and "llama-4" in str(self.ft_pytorch_model_path).lower()
        ):
            self.mlflow_hf_args["hf_pretrained_class"] = (
                "Llama4ForConditionalGeneration"
            )
        logger.info(f"Autoclasses for {self.mlflow_task_type} - {self.mlflow_hf_args}")

    def set_mlflow_model_parameters(self, model):
        """Prepare parameters for mlflow model."""
        self.mlflow_task_type = self.component_args.mlflow_task_type
        self.class_names = getattr(self.component_args, "class_names", None)
        self.model_name = self.component_args.model_name
        self._set_mlflow_hf_args()
        self.mlflow_ft_conf = self.ft_config.get("mlflow_ft_conf", {})
        self.mlflow_hftransformers_misc_conf = self.mlflow_ft_conf.get(
            "mlflow_hftransformers_misc_conf", {}
        )
        self.mlflow_model_signature = self.mlflow_ft_conf.get(
            "mlflow_model_signature", None
        )
        self.mlflow_save_model_kwargs = self.mlflow_ft_conf.get(
            "mlflow_save_model_kwargs", {}
        )
        self.metadata = getattr(
            self.component_args, ModelSelectorConstants.MODEL_METADATA, {}
        )
        self.mlflow_flavor = self.mlflow_ft_conf.get("mlflow_flavor", None)

        # tokenization parameters for inference
        # task related parameters
        mlflow_infer_params_file_path = str(
            Path(
                self.ft_pytorch_model_path,
                MLFlowHFFlavourConstants.INFERENCE_PARAMS_SAVE_NAME_WITH_EXT,
            )
        )
        with open(mlflow_infer_params_file_path, "r") as fp:
            mlflow_inference_params = json.load(fp)

        misc_conf = {
            MLFlowHFFlavourConstants.TASK_TYPE: self.mlflow_task_type,
            MLFlowHFFlavourConstants.TRAIN_LABEL_LIST: self.class_names,
            **mlflow_inference_params,
        }

        # if huggingface_id was passed, save it to MLModel file, otherwise not
        if self.model_name != MLFlowHFFlavourConstants.DEFAULT_MODEL_NAME:
            misc_conf.update({MLFlowHFFlavourConstants.HUGGINGFACE_ID: self.model_name})

        # auto classes need to be passed in misc_conf if custom code files are present in the model folder
        if hasattr(model.config, "auto_map"):
            misc_conf.update(self.mlflow_hf_args)
            logger.info(f"Updated misc conf with Auto classes - {misc_conf}")

        logger.info(
            f"Adding additional misc to MLModel - {self.mlflow_hftransformers_misc_conf}"
        )
        misc_conf = deep_update(misc_conf, self.mlflow_hftransformers_misc_conf)

        self.misc_conf = misc_conf
        logger.info(f"MlModel config - {self.misc_conf}")

        # Check if any code files are present in the model folder
        self.py_code_files = get_model_custom_code_files(
            self.ft_pytorch_model_path, model
        )

        # update metadata with mlflow task
        self.metadata = update_acft_metadata(
            metadata=self.metadata, finetuning_task=self.mlflow_task_type
        )

    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        model = load_model(
            self.ft_pytorch_model_path, self.component_args, self.ft_config
        )
        tokenizer = load_tokenizer(
            self.ft_pytorch_model_path, self.component_args, self.ft_config
        )
        return model, tokenizer

    def add_model_signature(self):
        """Add model signature to mlflow model."""
        if self.mlflow_model_signature is not None:
            logger.info(
                f"Adding mlflow model signature - {self.mlflow_model_signature}"
            )
            mlflow_model_file = Path(
                self.mlflow_model_save_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE
            )
            if mlflow_model_file.is_file():
                mlflow_model_data = {}
                with open(str(mlflow_model_file), "r") as fp:
                    yaml_data = yaml.safe_load(fp)
                    mlflow_model_data.update(yaml_data)
                    mlflow_model_data["signature"] = self.mlflow_model_signature

                with open(str(mlflow_model_file), "w") as fp:
                    yaml.dump(mlflow_model_data, fp)
                    logger.info(
                        f"Updated mlflow model file with 'signature': {self.mlflow_model_signature}"
                    )
            else:
                logger.info("No MLmodel file to update signature")
        else:
            logger.info("No signature will be added to mlflow model")


class Pytorch_to_HFTransformers_MlFlow_ModelConverter(
    ModelConverter, PyTorch_to_MlFlow_ModelConverter
):
    """Convert pytorch model to hftransformers mlflow model."""

    def __init__(self, component_args: Namespace) -> None:
        """Init."""
        super().__init__()
        super(ModelConverter, self).__init__(component_args)

    def convert_model(self) -> None:
        """Convert pytorch model to hftransformers mlflow model."""
        # load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        self.set_mlflow_model_parameters(model)

        # saving hftransformers mlflow model
        mlflow_model = Model(metadata=self.metadata)
        hf_mlflow.hftransformers.save_model(
            model,
            self.mlflow_model_save_path,
            tokenizer,
            model.config,
            self.misc_conf,
            code_paths=self.py_code_files,
            mlflow_model=mlflow_model,
            **self.mlflow_save_model_kwargs,
        )
        logger.info(f"Saved Finetuned Model with FLAVOR : {self.mlflow_flavor}")

        # copying the py files to mlflow destination
        # copy dynamic python files to config, model and tokenizer
        copy_code_files(
            self.py_code_files,
            [
                str(Path(self.mlflow_model_save_path, "data", dir))
                for dir in ["config", "model", "tokenizer"]
            ],
        )

        # saving additional details to mlflow model
        self.download_license_file(
            self.model_name, self.ft_pytorch_model_path, self.mlflow_model_save_path
        )
        self.add_model_signature()
        self.copy_finetune_config(
            self.ft_pytorch_model_path, self.mlflow_model_save_path
        )
        self.copy_ml_configs(self.ft_pytorch_model_path, self.mlflow_model_save_path)
        self.copy_preprocessor_config(
            self.ft_pytorch_model_path, self.mlflow_model_save_path
        )
        copy_tokenizer_files_to_model_folder(
            self.mlflow_model_save_path, self.component_args.task_name
        )

        logger.info("Saved MLFlow model using HFTransformers flavour.")


class Pytorch_to_OSS_MlFlow_ModelConverter(
    ModelConverter, PyTorch_to_MlFlow_ModelConverter
):
    """Convert pytorch model to OSS mlflow model."""

    def __init__(self, component_args: Namespace) -> None:
        """Init."""
        super().__init__()
        super(ModelConverter, self).__init__(component_args)

    @staticmethod
    def remove_unwanted_packages(model_save_path: str):
        """Remove unwanted packages from conda and requirements file."""
        if is_main_process():
            req_file_path = os.path.join(model_save_path, "requirements.txt")
            conda_file_path = os.path.join(model_save_path, "conda.yaml")
            requirements = None
            if os.path.exists(req_file_path):
                with open(req_file_path, "r") as f:
                    requirements = f.readlines()
                if requirements:
                    for package in UNWANTED_PACKAGES:
                        requirements = [
                            item
                            for item in requirements
                            if not item.startswith(package)
                        ]
                    logger.info("Updated requirements.txt file")

            conda_dict = None
            if os.path.exists(conda_file_path):
                with open(conda_file_path, "r") as f:
                    conda_dict = yaml.safe_load(f)
                if conda_dict is not None and "dependencies" in conda_dict:
                    for i in range(len(conda_dict["dependencies"])):
                        if "pip" in conda_dict["dependencies"][i] and isinstance(
                            conda_dict["dependencies"][i], dict
                        ):
                            pip_list = conda_dict["dependencies"][i]["pip"]
                            if len(pip_list) > 0:
                                for package in UNWANTED_PACKAGES:
                                    pip_list = [
                                        item
                                        for item in pip_list
                                        if not item.startswith(package)
                                    ]
                                conda_dict["dependencies"][i]["pip"] = pip_list
                                break
                    with open(conda_file_path, "w") as f:
                        yaml.safe_dump(conda_dict, f)
                    logger.info("Updated conda.yaml file")

    def is_t5_finetune(self, model_type) -> bool:
        """Check for t5 text-classification, translation, summarization."""
        return (
            self.component_args.task_name
            in [
                Tasks.SINGLE_LABEL_CLASSIFICATION,
                Tasks.TRANSLATION,
                Tasks.SUMMARIZATION,
            ]
            and model_type == HfModelTypes.T5
        )

    def convert_model(self) -> None:
        """Convert pytorch model to oss mlflow model."""
        # load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        self.set_mlflow_model_parameters(model)

        # Temp Fix:
        # specific check for t5 text-classification, translation, summarization so that base model
        # dependencies doesn't get pass and use transformers version 4.44.0 from infer dependencies
        if not self.is_t5_finetune(model.config.model_type):
            conda_file_path = Path(
                self.ft_pytorch_model_path, MLFlowHFFlavourConstants.CONDA_YAML_FILE
            )
            if conda_file_path.is_file():
                self.mlflow_save_model_kwargs.update(
                    {"conda_env": str(conda_file_path)}
                )
                logger.info(
                    f"Found {MLFlowHFFlavourConstants.CONDA_YAML_FILE} from base model. "
                    "Using it for saving Mlflow model."
                )

        # saving oss mlflow model
        model_pipeline = pipeline(
            task=self.mlflow_task_type,
            model=model,
            tokenizer=tokenizer,
            config=model.config,
        )
        mlflow.transformers.save_model(
            transformers_model=model_pipeline,
            path=self.mlflow_model_save_path,
            code_paths=self.py_code_files,
            metadata=self.metadata,
            **self.mlflow_save_model_kwargs,
        )
        logger.info(f"Saved Finetuned Model with FLAVOR : {self.mlflow_flavor}")

        # copying the py files to mlflow destination
        # copy dynamic python files to components and tokenizer
        copy_code_files(
            self.py_code_files,
            [
                str(Path(self.mlflow_model_save_path, dir))
                for dir in ["model", Path("components", "tokenizer")]
            ],
        )

        # saving additional details to mlflow model
        self.download_license_file(
            self.model_name, self.ft_pytorch_model_path, self.mlflow_model_save_path
        )
        self.add_model_signature()
        self.copy_finetune_config(
            self.ft_pytorch_model_path, self.mlflow_model_save_path
        )
        self.copy_ml_configs(self.ft_pytorch_model_path, self.mlflow_model_save_path)

        # Temp fix for t5 text-classification, translation, summarization
        if self.is_t5_finetune(model.config.model_type):
            self.remove_unwanted_packages(self.mlflow_model_save_path)

        logger.info("Saved MLFlow model using OSS flavour.")
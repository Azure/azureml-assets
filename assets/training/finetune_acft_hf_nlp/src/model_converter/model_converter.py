# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model converter component."""
from pathlib import Path
import argparse
import json
from argparse import Namespace
import yaml
import shutil

from abc import ABC, abstractmethod
from dataclasses import dataclass
from mlflow.models import Model

from azureml.acft.contrib.hf.nlp.utils.common_utils import deep_update
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.contrib.hf.nlp.constants.constants import SaveFileConstants, Tasks, MLFlowHFFlavourConstants
from azureml.acft.accelerator.constants import PeftLoRAConstants
from azureml.acft.accelerator.utils.code_utils import (
    get_model_custom_code_files,
    copy_code_files,
    update_json_file_and_overwrite,
)
from azureml.acft.accelerator.utils.license_utils import download_license_file
from azureml.acft.accelerator.utils.model_utils import print_model_summary

from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.mlflow_utils import update_acft_metadata
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME

from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

import azureml.evaluate.mlflow as mlflow

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.models.auto.modeling_auto import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
)

from peft import (
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForCausalLM,
)


logger = get_logger_app("azureml.acft.contrib.hf.scripts.components.scripts.model_converter.model_converter")

COMPONENT_NAME = "ACFT-Model_converter"


@dataclass
class ModelFormats:
    """Supported model formats."""

    PYTORCH = "pytorch"
    MLFLOW = "mlflow"


ACFT_TASKS_HUGGINGFACE_MODELS_MAPPING = {
    Tasks.SINGLE_LABEL_CLASSIFICATION: AutoModelForSequenceClassification,
    Tasks.MULTI_LABEL_CLASSIFICATION: AutoModelForSequenceClassification,
    Tasks.NAMED_ENTITY_RECOGNITION: AutoModelForTokenClassification,
    Tasks.SUMMARIZATION: AutoModelForSeq2SeqLM,
    Tasks.TRANSLATION: AutoModelForSeq2SeqLM,
    Tasks.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    Tasks.TEXT_GENERATION: AutoModelForCausalLM,
}


ACFT_TASKS_PEFT_MODELS_MAPPING = {
    Tasks.SINGLE_LABEL_CLASSIFICATION: AutoPeftModelForSequenceClassification,
    Tasks.MULTI_LABEL_CLASSIFICATION: AutoPeftModelForSequenceClassification,
    Tasks.NAMED_ENTITY_RECOGNITION: AutoPeftModelForTokenClassification,
    Tasks.SUMMARIZATION: AutoPeftModelForSeq2SeqLM,
    Tasks.TRANSLATION: AutoPeftModelForSeq2SeqLM,
    Tasks.QUESTION_ANSWERING: AutoPeftModelForQuestionAnswering,
    Tasks.TEXT_GENERATION: AutoPeftModelForCausalLM,
}


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model converter for hugging face models", allow_abbrev=False)

    parser.add_argument(
        "--output_dir",
        default="model_converter_output",
        type=str,
        help="folder to store model converter output",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="model path containing source model"
    )

    # currently indroducing to get already converted mlflow model from finetune
    parser.add_argument(
        "--converted_model",
        default=None,
        type=str,
        help="model path containing source model"
    )

    parser.add_argument(
        "--input_model_format",
        default=ModelFormats.PYTORCH,
        type=str,
        help="model format of input model"
    )
    parser.add_argument(
        "--output_model_format",
        default=ModelFormats.MLFLOW,
        type=str,
        help="model format of output model"
    )

    parser.add_argument(
        "--model_converter_config_path",
        default=None,
        type=str,
        help="model converter config file path"
    )

    return parser


def copy_finetune_args(args: Namespace) -> Namespace:
    """Copy finetune args to model converter."""
    # Read the finetune component args
    # Finetune Component + Preprocess Component + Model Selector Component ---> Model Converter Component
    # Since all Finetune component args are saved via Preprocess Component and Model Selector Component,
    # loading the Finetune args suffices
    finetune_args_load_path = str(Path(args.model_path, SaveFileConstants.FINETUNE_ARGS_SAVE_PATH))
    with open(finetune_args_load_path, 'r') as rptr:
        finetune_args = json.load(rptr)
        for key, value in finetune_args.items():
            if not hasattr(args, key):  # add keys that don't already exist
                setattr(args, key, value)

    finetune_config = {}
    finetune_config_load_path = str(Path(args.model_path, SaveFileConstants.ACFT_CONFIG_SAVE_PATH))
    if Path(finetune_config_load_path).is_file():
        with open(finetune_config_load_path, 'r') as rptr:
            finetune_config = json.load(rptr)

    setattr(args, "ft_config", finetune_config)

    return args


class ModelConverter(ABC):
    """Base model converter class."""

    def __init__(self) -> None:
        """Init."""
        pass

    @abstractmethod
    def convert_model(self, *args, **kwargs) -> None:
        """Convert model format."""
        pass


class Pytorch_to_MlFlow_ModelConverter(ModelConverter):
    """Convert Pytorch model to Mlflow model."""

    def __init__(self, component_args: Namespace) -> None:
        """Init."""
        super().__init__()
        self.component_args = component_args
        self.ft_config = component_args.ft_config
        ft_pytorch_model_path = component_args.model_path
        # the flag is to be deprecated in after migrating full model conversion logic to this script
        self.should_convert_model = False
        self.deepspeed_stage3_fted_model = False
        # currently only supporting deepspeed stage 3 + Peft LoRA model conversion
        # for others just copy model for now
        if Path(ft_pytorch_model_path, PeftLoRAConstants.PEFT_ADAPTER_WEIGHTS_FOLDER).is_dir() \
                and getattr(self.component_args, "is_deepspeed_zero3_enabled", False):
            self.should_convert_model = True
            self.deepspeed_stage3_fted_model = True

    def convert_model(self) -> None:
        """Convert pytorch model to mlflow model."""
        if self.should_convert_model:
            # as it is deepspeed stage 3 LoRA model we need to merge weights
            # and then convert to mlflow model
            # load the model in cpu
            ft_pytorch_model_path = self.component_args.model_path
            peft_lora_adapter_path = str(Path(ft_pytorch_model_path, PeftLoRAConstants.PEFT_ADAPTER_WEIGHTS_FOLDER))
            # copy PEFT_ADAPTER_WEIGHTS_FOLDER to ACFT_PEFT_CHECKPOINT_PATH and
            # change `base_model_path` in peft adapter_config to ft_pytorch_model_path
            if Path(PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH).is_dir():
                shutil.rmtree(PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH)
            # Path(PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH).mkdir(exist_ok=True, parents=True)
            shutil.copytree(peft_lora_adapter_path, PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH, dirs_exist_ok=True,)
            peft_lora_adapter_path = PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH
            peft_adapter_config_file = str(Path(
                peft_lora_adapter_path, PeftLoRAConstants.PEFT_ADAPTER_CONFIG_FILE_NAME
            ))
            update_config = {
                PeftLoRAConstants.PEFT_LORA_BASE_MODEL_PATH_KEY: ft_pytorch_model_path,
            }
            update_json_file_and_overwrite(peft_adapter_config_file, update_config)
            auto_cls = ACFT_TASKS_PEFT_MODELS_MAPPING[self.component_args.task_name]
            logger.info(f"Identified auto cls: {auto_cls}")
            load_model_kwargs = {
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
                "torch_dtype": "auto",
            }
            load_model_kwargs.update(self.ft_config.get("load_model_kwargs", {}))
            logger.info(f"Loading model with kwargs: {load_model_kwargs}")
            ds3_peft_model = auto_cls.from_pretrained(
                peft_lora_adapter_path,
                **load_model_kwargs,
            )  # loads both base weights and lora weights

            # merge LoRA weights
            merged_model = ds3_peft_model.merge_and_unload()
            print_model_summary(merged_model, True)

            # load tokenizer
            load_tokenizer_kwargs = self.ft_config.get("load_tokenizer_kwargs", {})
            logger.info(f"Loading tokenizer with kwargs: {load_model_kwargs}")
            tokenizer = AutoTokenizer.from_pretrained(
                ft_pytorch_model_path,
                **load_tokenizer_kwargs,
            )

            self.convert_to_mlflow_model(self.component_args, merged_model, tokenizer)
        else:
            ft_mlflow_model_path = self.component_args.converted_model
            if ft_mlflow_model_path and Path(ft_mlflow_model_path).is_dir():
                logger.info("Found existing FTed MLFlow model")
                shutil.copytree(ft_mlflow_model_path, self.component_args.output_dir, dirs_exist_ok=True,)
                logger.info("Copied existing MLFlow model")
            else:
                raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message="Should provide FTed MLFlow model path",
                    )
                )

    def convert_to_mlflow_model(
        self,
        component_args: Namespace,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """Convert to mlflow model."""
        mlflow_model_save_path = component_args.output_dir
        pytorch_model_path = component_args.model_path
        mlflow_infer_params_file_path = str(Path(
            pytorch_model_path, MLFlowHFFlavourConstants.INFERENCE_PARAMS_SAVE_NAME_WITH_EXT
        ))
        mlflow_task_type = component_args.mlflow_task_type
        class_names = getattr(component_args, "class_names", None)
        model_name = component_args.model_name
        model_name_or_path = pytorch_model_path
        mlflow_hf_args = {
            "hf_config_class": "AutoConfig",
            "hf_tokenizer_class": "AutoTokenizer",
            "hf_pretrained_class": "AutoModelForCausalLM",
        }
        mlflow_ft_conf = self.ft_config.get("mlflow_ft_conf", {})
        mlflow_hftransformers_misc_conf = mlflow_ft_conf.get("mlflow_hftransformers_misc_conf", {})
        mlflow_model_signature = mlflow_ft_conf.get("mlflow_model_signature", None)
        mlflow_save_model_kwargs = mlflow_ft_conf.get("mlflow_save_model_kwargs", {})

        # tokenization parameters for inference
        # task related parameters
        with open(mlflow_infer_params_file_path, 'r') as fp:
            mlflow_inference_params = json.load(fp)

        misc_conf = {
            MLFlowHFFlavourConstants.TASK_TYPE: mlflow_task_type,
            MLFlowHFFlavourConstants.TRAIN_LABEL_LIST: class_names,
            **mlflow_inference_params,
        }

        # if huggingface_id was passed, save it to MLModel file, otherwise not
        if model_name != MLFlowHFFlavourConstants.DEFAULT_MODEL_NAME:
            misc_conf.update({MLFlowHFFlavourConstants.HUGGINGFACE_ID: model_name})

        # auto classes need to be passed in misc_conf if custom code files are present in the model folder
        if hasattr(model.config, "auto_map"):
            misc_conf.update(mlflow_hf_args)
            logger.info(f"Updated misc conf with Auto classes - {misc_conf}")

        logger.info(f"Adding additional misc to MLModel - {mlflow_hftransformers_misc_conf}")
        misc_conf = deep_update(misc_conf, mlflow_hftransformers_misc_conf)

        # Check if any code files are present in the model folder
        py_code_files = get_model_custom_code_files(model_name_or_path, model)

        # adding metadata to mlflow model
        metadata = self.component_args.model_metadata
        metadata = update_acft_metadata(metadata=metadata, finetuning_task=mlflow_task_type)
        mlflow_model = Model(metadata=metadata)

        mlflow.hftransformers.save_model(
            model,
            mlflow_model_save_path,
            tokenizer,
            model.config,
            misc_conf,
            code_paths=py_code_files,
            mlflow_model=mlflow_model,
            **mlflow_save_model_kwargs,
        )

        # copying the py files to mlflow destination
        # copy dynamic python files to config, model and tokenizer
        copy_code_files(
            py_code_files,
            [str(Path(mlflow_model_save_path, 'data', dir)) for dir in ['config', 'model', 'tokenizer']]
        )

        # save LICENSE file to MlFlow model
        if model_name_or_path:
            license_file_path = Path(model_name_or_path, MLFlowHFFlavourConstants.LICENSE_FILE)
            if license_file_path.is_file():
                shutil.copy(str(license_file_path), mlflow_model_save_path)
                logger.info("LICENSE file is copied to mlflow model folder")
            else:
                download_license_file(model_name, str(mlflow_model_save_path))

        # setting mlflow model signature for inference
        if mlflow_model_signature is not None:
            logger.info(f"Adding mlflow model signature - {mlflow_model_signature}")
            mlflow_model_file = Path(mlflow_model_save_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
            if mlflow_model_file.is_file():
                mlflow_model_data = {}
                with open(str(mlflow_model_file), "r") as fp:
                    yaml_data = yaml.safe_load(fp)
                    mlflow_model_data.update(yaml_data)
                    mlflow_model_data["signature"] = mlflow_model_signature

                with open(str(mlflow_model_file), "w") as fp:
                    yaml.dump(mlflow_model_data, fp)
                    logger.info(f"Updated mlflow model file with 'signature': {mlflow_model_signature}")
            else:
                logger.info("No MLmodel file to update signature")
        else:
            logger.info("No signature will be added to mlflow model")

        logger.info("Saved as mlflow model at {}".format(mlflow_model_save_path))


def model_converter(args: Namespace):
    """Model convertor."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # additional logging
    logger.info(f"Model name: {getattr(args, 'model_name', None)}")
    logger.info(f"Task name: {getattr(args, 'task_name', None)}")

    input_model_format = args.input_model_format
    output_model_format = args.output_model_format

    if input_model_format == ModelFormats.PYTORCH and output_model_format == ModelFormats.MLFLOW:
        model_converter = Pytorch_to_MlFlow_ModelConverter(args)
    else:
        raise NotImplementedError(
            f"Conversion of model from {input_model_format} format "
            f"to {output_model_format} format is not supported."
        )

    model_converter.convert_model()

    logger.info(f"Successfully converted {input_model_format} model to {output_model_format} model")


@swallow_all_exceptions(time_delay=60)
def main():
    """Parse args and import model."""
    # args
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Copy the args generated in the finetune step
    args = copy_finetune_args(args)

    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    model_converter(args)


if __name__ == "__main__":
    main()

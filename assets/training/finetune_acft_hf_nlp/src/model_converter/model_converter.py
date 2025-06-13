# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model converter component."""
from pathlib import Path
import argparse
import json
from argparse import Namespace

from dataclasses import dataclass
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.contrib.hf.nlp.constants.constants import SaveFileConstants, MLFLOW_FLAVORS, HfModelTypes

from model_converter_adapters import (
    Pytorch_to_HFTransformers_MlFlow_ModelConverter,
    Pytorch_to_OSS_MlFlow_ModelConverter,
)

from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME


logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.model_converter.model_converter")


COMPONENT_NAME = "ACFT-Model_converter"

REFINED_WEB = "RefinedWeb"
MIXFORMER_SEQUENTIAL = "mixformer-sequential"  # Phi models


@dataclass
class ModelFormats:
    """Supported model formats."""

    PYTORCH = "pytorch"
    MLFLOW = "mlflow"


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model converter for hugging face models", allow_abbrev=False)

    parser.add_argument(
        "--output_dir",
        default="mlflow_model_folder",
        type=str,
        help="Output dir to save the finetune model as mlflow model.",
    )

    parser.add_argument(
        "--model_path",
        default="pytorch_model_folder",
        type=str,
        help="Model path containing finetuned model, configs, tokenizer and checkpoints."
    )

    parser.add_argument(
        "--model_import_output",
        default=None,
        type=str,
        help=(
            "output folder of model selector containing model configs, tokenizer, checkpoints in case of model_id."
            "If huggingface_id is selected, the model download happens dynamically on the fly"
        ),
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


def update_lora_target_modules():
    """Update peft config with falcon target layers."""
    import peft

    models_to_lora_target_modules_map = getattr(
        peft.utils.other, "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING", {}
    )
    models_to_lora_target_modules_map.update(
        {
            HfModelTypes.REFINEDWEBMODEL: ["query_key_value"],
            REFINED_WEB: ["query_key_value"],
            HfModelTypes.FALCON: ["query_key_value"],
        }
    )
    setattr(
        peft.utils.other,
        "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
        models_to_lora_target_modules_map
    )
    setattr(
        peft.tuners.lora,
        "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
        models_to_lora_target_modules_map
    )
    logger.info(
        f"Updated lora target modules map: {peft.tuners.lora.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING}")


def model_converter(args: Namespace):
    """Model convertor."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # additional logging
    logger.info(f"Model name: {getattr(args, 'model_name', None)}")
    logger.info(f"Task name: {getattr(args, 'task_name', None)}")

    input_model_format = args.input_model_format
    output_model_format = args.output_model_format

    if input_model_format == ModelFormats.PYTORCH and output_model_format == ModelFormats.MLFLOW:
        mlflow_model_flavour = getattr(args, 'mlflow_flavor', None)
        logger.info(f"MLFlow flavour: {mlflow_model_flavour}")
        if mlflow_model_flavour == MLFLOW_FLAVORS.TRANSFORMERS:
            model_converter = Pytorch_to_OSS_MlFlow_ModelConverter(args)
        else:
            model_converter = Pytorch_to_HFTransformers_MlFlow_ModelConverter(args)
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

    # XXX Hack to support model loading in accelerator package for falcon models
    # with `trust_remote_code=True`
    # This is needed as FT config is ONLY used in contrib package which is causing
    # failure when loading the model in accelerator as part of peft weights merge
    if hasattr(args, "model_type") and args.model_type in [
        HfModelTypes.REFINEDWEBMODEL,
        HfModelTypes.FALCON,
        REFINED_WEB,
        MIXFORMER_SEQUENTIAL
    ]:
        from functools import partial
        from transformers.models.auto import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
            AutoModelForCausalLM,
        )

        # Updata lora target modules for falcon
        update_lora_target_modules()

        AutoModelForSequenceClassification.from_pretrained = partial(
            AutoModelForSequenceClassification.from_pretrained, trust_remote_code=True
        )
        AutoModelForTokenClassification.from_pretrained = partial(
            AutoModelForTokenClassification.from_pretrained, trust_remote_code=True
        )
        AutoModelForQuestionAnswering.from_pretrained = partial(
            AutoModelForQuestionAnswering.from_pretrained, trust_remote_code=True
        )
        AutoModelForCausalLM.from_pretrained = partial(
            AutoModelForCausalLM.from_pretrained, trust_remote_code=True
        )
        logger.info("Updated `from_pretrained` method for Seq cls, Tok cls, QnA and Text Gen")

    model_converter(args)


if __name__ == "__main__":
    main()

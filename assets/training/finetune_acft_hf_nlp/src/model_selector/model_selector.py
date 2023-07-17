# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model selector component."""
from pathlib import Path
import argparse
import json
from argparse import Namespace
import copy

from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.contrib.hf.nlp.constants.constants import SaveFileConstants

from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME


logger = get_logger_app("azureml.acft.contrib.hf.scripts.components.scripts.model_selector.model_selector")

COMPONENT_NAME = "ACFT-Model_import"

# user config passed along with model, will be preffered over default settings
ACFT_CONFIG = {
    "tiiuae/falcon-7b": {
        "load_config_kwargs": {
            "trust_remote_code": True,
        },
        "load_tokenizer_kwargs": {
            # adding eos_token as pad_token. The value of eos_token is taken from tokenization_config.json file
            "pad_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
        "load_model_kwargs": {
            "trust_remote_code": True,
        },
        "finetune_args": {},
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "config_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "model_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "tokenizer_config": {
                    "return_token_type_ids": False,
                },
            },
            "mlflow_save_model_kwargs": {
                "extra_pip_requirements": ["einops"],
            },
        },
    },
    "tiiuae/falcon-40b": {
        "load_config_kwargs": {
            "trust_remote_code": True,
        },
        "load_tokenizer_kwargs": {
            # adding eos_token as pad_token. The value of eos_token is taken from tokenization_config.json file
            "pad_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
        "load_model_kwargs": {
            "trust_remote_code": True,
        },
        "finetune_args": {},
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "config_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "model_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "tokenizer_config": {
                    "return_token_type_ids": False,
                },
            },
            "mlflow_save_model_kwargs": {
                "extra_pip_requirements": ["einops"],
            },
        },
    },
}


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model selector for hugging face models", allow_abbrev=False)

    parser.add_argument(
        "--output_dir",
        default="model_selector_output",
        type=str,
        help="folder to store model selector outputs",
    )

    parser.add_argument(
        "--huggingface_id",
        default=None,
        type=str,
        help="Input HuggingFace model id takes priority over model_id.",
    )

    # Task settings
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model id used to load model checkpoint.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="SingleLabelClassification",
        help="Task Name",
    )

    # Continual Finetuning
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        help="model path containing pytorch model"
    )
    parser.add_argument(
        "--mlflow_model_path",
        default=None,
        type=str,
        help="model path containing mlflow model"
    )

    # FT config
    parser.add_argument(
        "--finetune_config_path",
        default=None,
        type=str,
        help="finetune config file path"
    )

    return parser


def model_selector(args: Namespace):
    """Model selector."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.huggingface_id is not None:
        # remove the spaces at either ends of hf id
        args.model_name = args.huggingface_id.strip()
    else:
        # TODO Revist whether `model_id` is still relevant
        args.model_name = args.model_id

    task_runner = get_task_runner(task_name=args.task_name)()
    task_runner.run_modelselector(**vars(args))

    # for base curated models forward MLmodel info
    if getattr(args, "mlflow_model_path", None) is not None:
        import shutil
        from azureml.acft.contrib.hf.nlp.constants.constants import MLFlowHFFlavourConstants
        mlflow_config_file = Path(args.mlflow_model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
        if mlflow_config_file.is_file():
            shutil.copy(str(mlflow_config_file), args.output_dir)
            logger.info("Copied MLmodel file to output dir")
        else:
            logger.info("MLmodel file does not exist")
    else:
        logger.info("mlflow_model_path is empty")

    # load user provided finetune_config.json
    ft_config_path = getattr(args, "finetune_config_path", None)
    # if ft_config_path is not provided check inside pytorch/mlflow model folder
    if ft_config_path is None:
        model_path = getattr(args, "pytorch_model_path", None) \
            if getattr(args, "pytorch_model_path", None) is not None else getattr(args, "mlflow_model_path", None)
        if model_path is not None:
            ft_model_config_path = Path(model_path, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)
            if ft_model_config_path.is_file():
                ft_config_path = str(ft_model_config_path)
    if ft_config_path is not None:
        try:
            with open(ft_config_path, "r") as rptr:
                ft_config_data = json.load(rptr)
        except:
            logger.info(f"Unable to load {SaveFileConstants.ACFT_CONFIG_SAVE_PATH}")
            ft_config_data = {}
    else:
        logger.info(f"{SaveFileConstants.ACFT_CONFIG_SAVE_PATH} does not exist")
        ft_config_data = {}

    # fetch model_name
    model_name = None
    try:
        model_selector_file = Path(args.output_dir, SaveFileConstants.MODEL_SELECTOR_ARGS_SAVE_PATH)
        with open(model_selector_file, "r") as rptr:
            data = json.load(rptr)
            model_name = data.get("model_name")
            logger.info(f"Fetched model name - {model_name}")
    except:
        logger.info(f"Unable to read model_name from {SaveFileConstants.MODEL_SELECTOR_ARGS_SAVE_PATH}")

    if model_name is not None and model_name in ACFT_CONFIG:
        model_ft_config = copy.deepcopy(ACFT_CONFIG[model_name])
        model_ft_config.update(ft_config_data)
        ft_config_data = copy.deepcopy(model_ft_config)
        logger.info(f"Updated FT config data - {ft_config_data}")
    else:
        logger.info(f"Not updating FT config data - {ft_config_data}")

    # saving FT config
    with open(str(Path(args.output_dir, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)), "w") as rptr:
        json.dump(ft_config_data, rptr, indent=2)
    logger.info(f"Saved {SaveFileConstants.ACFT_CONFIG_SAVE_PATH}")

    # additional logging
    logger.info(f"Model name: {getattr(args, 'model_name', None)}")
    logger.info(f"Task name: {getattr(args, 'task_name', None)}")


@swallow_all_exceptions(time_delay=60)
def main():
    """Parse args and import model."""
    # args
    parser = get_parser()
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    model_selector(args)


if __name__ == "__main__":
    main()

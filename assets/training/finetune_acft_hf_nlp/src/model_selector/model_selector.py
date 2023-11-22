# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model selector component."""
import shutil
from pathlib import Path
import json
import argparse
from argparse import Namespace
import copy
import yaml
from typing import Dict, Any, Optional

from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.utils.common_utils import deep_update
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.contrib.hf.nlp.constants.constants import SaveFileConstants, MLFlowHFFlavourConstants

from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME

from finetune_config import FinetuneConfig


logger = get_logger_app("azureml.acft.contrib.hf.scripts.components.scripts.model_selector.model_selector")

COMPONENT_NAME = "ACFT-Model_import"

# TODO - Move REFINED_WEB to :dataclass HfModelTypes
REFINED_WEB = "RefinedWeb"
MIXFORMER_SEQUENTIAL = "mixformer-sequential"  # Phi models


# TODO Move this constants class to package
class ModelSelectorConstants:
    """Model import constants."""

    ASSET_ID_NOT_FOUND = "ASSET_ID_NOT_FOUND",
    MODEL_NAME_NOT_FOUND = "MODEL_NAME_NOT_FOUND"


def get_model_asset_id() -> str:
    """Read the model asset id from the run context."""
    try:
        from azureml.core import Run

        run_ctx = Run.get_context()
        if isinstance(run_ctx, Run):
            run_details = run_ctx.get_details()
            return run_details['runDefinition']['inputAssets']['mlflow_model_path']['asset']['assetId']
        else:
            logger.info("Found offline run")
            return ModelSelectorConstants.ASSET_ID_NOT_FOUND
    except Exception as e:
        logger.info(f"Could not fetch the model asset id: {e}")
        return ModelSelectorConstants.ASSET_ID_NOT_FOUND


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


def model_selector(args: Namespace) -> Dict[str, Any]:
    """Model selector main.

    :param args - component args
    :type Namespace
    :return Meta data saved by model selector component (model selector args)
    :rtype Dict[str, Any]
    """
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.huggingface_id is not None:
        # remove the spaces at either ends of hf id
        args.model_name = args.huggingface_id.strip()
    else:
        # TODO Revist whether `model_id` is still relevant
        args.model_name = args.model_id

    # Add the model asset id to model_selector_args
    model_asset_id = get_model_asset_id()
    logger.info(f"Model asset id: {model_asset_id}")
    setattr(args, "model_asset_id", model_asset_id)

    task_runner = get_task_runner(task_name=args.task_name)()
    task_runner.run_modelselector(**vars(args))

    # read model selector args
    # fetch model details
    model_selector_args_save_path = Path(args.output_dir, SaveFileConstants.MODEL_SELECTOR_ARGS_SAVE_PATH)
    with open(model_selector_args_save_path, "r") as rptr:
        model_selector_args = json.load(rptr)

    return model_selector_args


def fetch_model_type(model_path: str) -> Optional[str]:
    """Fetch the model type.

    :param model_path - path to the model artifacts
    :type str
    :return model type
    :rtype Optional[str]
    """
    model_type = None
    try:
        # fetch model_type
        model_config_path = Path(model_path, "config.json")
        if model_config_path.is_file():
            with open(model_config_path, "r") as fp:
                model_config = json.load(fp)
                model_type = model_config.get("model_type", None)
        else:
            logger.info(f"Model config.json does not exist for {model_path}")
    except Exception:
        logger.info(f"Unable to fetch model_type for {model_path}")

    return model_type


def read_base_model_finetune_config(mlflow_model_path: str, task_name: str) -> Dict[str, Any]:
    """Read the finetune config from base model.

    :param mlflow_model_path - Path to the mlflow model
    :type str
    :param task_name - Finetune task
    :type str
    :return base model finetune config
    :type Optional[Dict[str, Any]]
    """
    if mlflow_model_path is None:
        return {}

    mlflow_config_file = Path(mlflow_model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
    mlflow_ftconf_data = {}
    if mlflow_config_file.is_file():
        # pass mlflow data to ft config if available
        mlflow_data = None
        try:
            with open(str(mlflow_config_file), "r") as fp:
                mlflow_data = yaml.safe_load(fp)
            if mlflow_data and "flavors" in mlflow_data:
                for key in mlflow_data["flavors"]:
                    if key in ["hftransformers", "hftransformersv2"]:
                        for key2 in mlflow_data["flavors"][key]:
                            if key2 == "generator_config" and task_name == "TextGeneration":
                                generator_config = mlflow_data["flavors"][key]["generator_config"]
                                mlflow_ftconf_data_temp = {
                                        "load_config_kwargs": copy.deepcopy(generator_config),
                                        "mlflow_ft_conf": {
                                            "mlflow_hftransformers_misc_conf": {
                                                "generator_config": copy.deepcopy(generator_config),
                                            },
                                        },
                                    }
                                mlflow_ftconf_data = deep_update(mlflow_ftconf_data_temp, mlflow_ftconf_data)
                            elif key2 == "model_hf_load_kwargs":
                                model_hf_load_kwargs = mlflow_data["flavors"][key]["model_hf_load_kwargs"]
                                mlflow_ftconf_data_temp = {
                                        "mlflow_ft_conf": {
                                            "mlflow_hftransformers_misc_conf": {
                                                "model_hf_load_kwargs": copy.deepcopy(model_hf_load_kwargs),
                                            },
                                        },
                                    }
                                mlflow_ftconf_data = deep_update(mlflow_ftconf_data_temp, mlflow_ftconf_data)
        except Exception:
            logger.info("Error while updating base model finetune config from MlModel file.")
    else:
        logger.info("MLmodel file does not exist")

    return mlflow_ftconf_data


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

    # run model selector
    model_selector_args = model_selector(args)
    model_name = model_selector_args.get("model_name", ModelSelectorConstants.MODEL_NAME_NOT_FOUND)
    logger.info(f"Model name - {model_name}")
    logger.info(f"Task name: {getattr(args, 'task_name', None)}")

    # load ft config and update ACFT config
    # finetune_config_dict = load_finetune_config(args)
    ft_config_obj = FinetuneConfig(
        task_name=args.task_name,
        model_name=model_name,
        model_type=fetch_model_type(str(Path(args.output_dir, model_name))),
        artifacts_finetune_config_path=str(
            Path(
                args.pytorch_model_path or args.mlflow_model_path,
                SaveFileConstants.ACFT_CONFIG_SAVE_PATH
            )
        ),
        io_finetune_config_path=args.finetune_config_path
    )
    finetune_config = ft_config_obj.get_finetune_config()

    # read finetune config from base mlmodel file
    # Priority order: io_finetune_config > artifacts_finetune_config > base_model_finetune_config
    updated_finetune_config = deep_update(
        read_base_model_finetune_config(
            args.mlflow_model_path,
            args.task_name
        ),
        finetune_config
    )
    logger.info(f"Updated finetune config with base model config: {updated_finetune_config}")

    # copy the mlmodel file to output dir
    mlflow_config_file = Path(args.mlflow_model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
    if mlflow_config_file.is_file():
        shutil.copy(str(mlflow_config_file), args.output_dir)
        logger.info("Copied MLmodel file to output dir")

    # save FT config
    with open(str(Path(args.output_dir, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)), "w") as rptr:
        json.dump(updated_finetune_config, rptr, indent=2)
    logger.info(f"Saved {SaveFileConstants.ACFT_CONFIG_SAVE_PATH}")


if __name__ == "__main__":
    main()

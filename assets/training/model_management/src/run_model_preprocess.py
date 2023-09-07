# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model preprocessor module."""

import argparse
import os
import json
import shutil
from azureml.model.mgmt.config import AppName, ModelFlavor
from azureml.model.mgmt.processors.transformers.config import HF_CONF
from azureml.model.mgmt.processors.preprocess import run_preprocess
from azureml.model.mgmt.processors.transformers.config import SupportedTasks
from azureml.model.mgmt.processors.pyfunc.vision.config import Tasks
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions, UnsupportedTaskType
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from pathlib import Path
from tempfile import TemporaryDirectory


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.CONVERT_MODEL_TO_MLFLOW


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=False, help="Hugging Face model ID")
    parser.add_argument("--task-name", type=str, required=False, help="Hugging Face task type")
    parser.add_argument("--hf-config-args", type=str, required=False, help="Hugging Face config init args")
    parser.add_argument("--hf-tokenizer-args", type=str, required=False, help="Hugging Face tokenizer init args")
    parser.add_argument("--hf-model-args", type=str, required=False, help="Hugging Face model init args")
    parser.add_argument("--hf-pipeline-args", type=str, required=False, help="Hugging Face pipeline init args")
    parser.add_argument("--hf-config-class", type=str, required=False, help="Hugging Face config class")
    parser.add_argument("--hf-model-class", type=str, required=False, help="Hugging Face model class ")
    parser.add_argument("--hf-tokenizer-class", type=str, required=False, help="Hugging tokenizer class")
    # argparse issue: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument(
        "--hf-use-experimental-features",
        type=str,
        default="false",
        help="Enable experimental features for hugging face MLflow model conversion",
        required=False,
    )

    parser.add_argument(
        "--extra-pip-requirements",
        type=str,
        required=False,
        help="Extra pip dependecies which is not present in current env but needed to load model env.",
    )

    parser.add_argument(
        "--mlflow-flavor",
        type=str,
        default=ModelFlavor.TRANSFORMERS.value,
        help="Model flavor",
    )
    parser.add_argument(
        "--model-download-metadata",
        type=Path,
        required=False,
        help="Model download details",
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--license-file-path", type=Path, required=False, help="License file path")
    parser.add_argument(
        "--mlflow-model-output-dir",
        type=Path,
        required=True,
        help="Output MLflow model",
    )
    parser.add_argument(
        "--model-import-job-path",
        type=Path,
        required=True,
        help="JSON file containing model job path for model lineage",
    )
    return parser


def _validate_transformers_args(args):
    if not args.get("model_id"):
        raise Exception("model_id is a required parameter for hftransformers MLflow flavor.")
    if not args.get("task"):
        raise Exception("task is a required parameter for hftransformers MLflow flavor.")
    task = args["task"]
    if not SupportedTasks.has_value(task):
        raise Exception(f"Unsupported task {task} for hftransformers MLflow flavor.")


def _validate_pyfunc_args(pyfunc_args):
    if not pyfunc_args.get("task"):
        raise Exception("task is a required parameter for pyfunc flavor.")
    task = pyfunc_args["task"]
    if not Tasks.has_value(task):
        raise Exception(f"Unsupported task {task} for pyfunc flavor.")


@swallow_all_exceptions(logger)
def run():
    """Run preprocess."""
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_id = args.model_id
    task_name = args.task_name
    mlflow_flavor = args.mlflow_flavor
    hf_config_args = args.hf_config_args
    hf_tokenizer_args = args.hf_tokenizer_args
    hf_model_args = args.hf_model_args
    hf_pipeline_args = args.hf_pipeline_args
    hf_config_class = args.hf_config_class
    hf_model_class = args.hf_model_class
    hf_tokenizer_class = args.hf_tokenizer_class
    hf_use_experimental_features = False if args.hf_use_experimental_features.lower() == "false" else True
    extra_pip_requirements = args.extra_pip_requirements

    model_download_metadata_path = args.model_download_metadata
    model_path = args.model_path
    mlflow_model_output_dir = args.mlflow_model_output_dir
    model_import_job_path = args.model_import_job_path
    license_file_path = args.license_file_path

    if not ModelFlavor.has_value(mlflow_flavor):
        raise Exception(f"Unsupported model flavor {mlflow_flavor}")

    preprocess_args = {}
    if model_download_metadata_path:
        with open(model_download_metadata_path) as f:
            download_details = json.load(f)
            preprocess_args.update(download_details.get("tags", {}))
            preprocess_args.update(download_details.get("properties", {}))
            preprocess_args["misc"] = download_details.get("misc", [])

    if task_name is None or not SupportedTasks.has_value(task_name.lower()):
        task_name = preprocess_args.get("task")
        logger.warning("task_name is not provided or not supported. "
                       f"Using task_name={task_name} from model download metadata.")

    if task_name is None:
        raise AzureMLException._with_error(
                AzureMLError.create(UnsupportedTaskType, task_type=args.task_name,
                                    supported_tasks=SupportedTasks.list_values())
            )

    preprocess_args["task"] = task_name.lower()
    preprocess_args["model_id"] = model_id if model_id else preprocess_args.get("model_id")
    preprocess_args[HF_CONF.EXTRA_PIP_REQUIREMENTS.value] = extra_pip_requirements
    preprocess_args[HF_CONF.HF_CONFIG_ARGS.value] = hf_config_args
    preprocess_args[HF_CONF.HF_TOKENIZER_ARGS.value] = hf_tokenizer_args
    preprocess_args[HF_CONF.HF_MODEL_ARGS.value] = hf_model_args
    preprocess_args[HF_CONF.HF_PIPELINE_ARGS.value] = hf_pipeline_args
    preprocess_args[HF_CONF.HF_CONFIG_CLASS.value] = hf_config_class
    preprocess_args[HF_CONF.HF_PRETRAINED_CLASS.value] = hf_model_class
    preprocess_args[HF_CONF.HF_TOKENIZER_CLASS.value] = hf_tokenizer_class
    preprocess_args[HF_CONF.HF_USE_EXPERIMENTAL_FEATURES.value] = hf_use_experimental_features

    # update custom dimensions with input parameters
    custom_dimensions.update_custom_dimensions(preprocess_args)

    logger.info(f"Preprocess args : {preprocess_args}")

    # TODO: move validations to respective convertors
    if mlflow_flavor == ModelFlavor.TRANSFORMERS.value:
        _validate_transformers_args(preprocess_args)
    elif mlflow_flavor == ModelFlavor.PYFUNC.value:
        _validate_pyfunc_args(preprocess_args)

    with TemporaryDirectory(dir=mlflow_model_output_dir) as working_dir, TemporaryDirectory(
        dir=mlflow_model_output_dir
    ) as temp_dir:
        run_preprocess(mlflow_flavor, model_path, working_dir, temp_dir, **preprocess_args)
        shutil.copytree(working_dir, mlflow_model_output_dir, dirs_exist_ok=True)

    # Copy license file to output model path
    if license_file_path:
        shutil.copy(license_file_path, mlflow_model_output_dir)

    logger.info(f"listing output directory files: {mlflow_model_output_dir}:\n{os.listdir(mlflow_model_output_dir)}")

    # Add job path
    this_job = os.environ["MLFLOW_RUN_ID"]
    path = f"azureml://jobs/{this_job}/outputs/mlflow_model_folder"
    model_path_dict = {"path": path}
    json_object = json.dumps(model_path_dict, indent=4)
    with open(model_import_job_path, "w") as outfile:
        outfile.write(json_object)
    logger.info("Finished writing job path")


if __name__ == "__main__":
    run()

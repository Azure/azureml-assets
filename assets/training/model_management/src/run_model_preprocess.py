# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model preprocessor module."""

import argparse
import os
import json
import shutil
from azureml.model.mgmt.config import AppName, ModelFramework
from azureml.model.mgmt.processors.transformers.config import HF_CONF
from azureml.model.mgmt.processors.preprocess import run_preprocess, check_for_py_files
from azureml.model.mgmt.processors.transformers.config import SupportedTasks as TransformersSupportedTasks
from azureml.model.mgmt.processors.pyfunc.config import SupportedTasks as PyFuncSupportedTasks
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions, ModelImportErrorStrings
from azure.ai.ml.exceptions import ErrorTarget, ErrorCategory, MlException
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from pathlib import Path
from tempfile import TemporaryDirectory


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.CONVERT_MODEL_TO_MLFLOW


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=False, help="Hugging Face model ID")
    parser.add_argument("--task-name", type=str, required=False, help="Hugging Face task type")
    parser.add_argument("--model-flavor", type=str, required=False, help="Model flavor HFtransformersV2 / OSS")
    parser.add_argument("--vllm-enabled", type=str, required=True, default="false", help="Flag to enabled vllm")
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
        "--inference-base-image",
        type=str,
        required=False,
        help="The azureml base image to use in model inference. \
            This ACR id is added in metadata of Mlmodel (mlflow)."
    )

    parser.add_argument(
        "--model-framework",
        type=str,
        default=ModelFramework.HUGGINGFACE.value,
        help="Model framework",
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
    return parser


@swallow_all_exceptions(logger)
def run():
    """Run preprocess."""
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_id = args.model_id
    task_name = args.task_name
    model_flavor = args.model_flavor
    vllm_enabled = False if args.vllm_enabled.lower() == "false" else True
    model_framework = args.model_framework
    hf_config_args = args.hf_config_args
    hf_tokenizer_args = args.hf_tokenizer_args
    hf_model_args = args.hf_model_args
    hf_pipeline_args = args.hf_pipeline_args
    hf_config_class = args.hf_config_class
    hf_model_class = args.hf_model_class
    hf_tokenizer_class = args.hf_tokenizer_class
    hf_use_experimental_features = False if args.hf_use_experimental_features.lower() == "false" else True
    extra_pip_requirements = args.extra_pip_requirements
    inference_base_image = args.inference_base_image

    model_download_metadata_path = args.model_download_metadata
    model_path = args.model_path
    mlflow_model_output_dir = args.mlflow_model_output_dir
    license_file_path = args.license_file_path
    TRUST_CODE_KEY = "trust_remote_code=True"

    if not ModelFramework.has_value(model_framework):
        raise Exception(f"Unsupported model framework {model_framework}")

    preprocess_args = {}
    if model_download_metadata_path:
        with open(model_download_metadata_path) as f:
            download_details = json.load(f)
            preprocess_args.update(download_details.get("tags", {}))
            preprocess_args.update(download_details.get("properties", {}))
            preprocess_args["misc"] = download_details.get("misc", [])

    if task_name is None or \
            (
                not TransformersSupportedTasks.has_value(task_name.lower()) and
                not PyFuncSupportedTasks.has_value(task_name.lower())
            ):
        task_name = preprocess_args.get("task")
        logger.warning("task_name is not provided or not supported. "
                       f"Using task_name={task_name} from model download metadata.")

    if task_name is None:
        supported_tasks = set(TransformersSupportedTasks.list_values() + PyFuncSupportedTasks.list_values())
        message = ModelImportErrorStrings.UNSUPPORTED_TASK_TYPE.format(
            task_type=args.task_name, supported_tasks=list(supported_tasks)
        )
        raise MlException(
            message=message, no_personal_data_message=message,
            error_category=ErrorCategory.USER_ERROR, target=ErrorTarget.COMPONENT
        )

    files = check_for_py_files(model_path)
    logger.info(f"check if model folder contains .py files or not: {files}")
    if files:
        if hf_model_args is None or TRUST_CODE_KEY not in hf_model_args:
            hf_model_args = TRUST_CODE_KEY
            logger.warning(f"{TRUST_CODE_KEY} is not provided for hf_model_args. Using {TRUST_CODE_KEY}.")
        if hf_tokenizer_args is None or TRUST_CODE_KEY not in hf_tokenizer_args:
            hf_tokenizer_args = TRUST_CODE_KEY
            logger.warning(f"{TRUST_CODE_KEY} is not provided for hf_tokenizer_args. Using {TRUST_CODE_KEY}.")
        if hf_config_args is None or TRUST_CODE_KEY not in hf_config_args:
            hf_config_args = TRUST_CODE_KEY
            logger.warning(f"{TRUST_CODE_KEY} is not provided for hf_config_args. Using {TRUST_CODE_KEY}.")
    preprocess_args["task"] = task_name.lower()
    preprocess_args["model_id"] = model_id if model_id else preprocess_args.get("model_id")
    preprocess_args["model_flavor"] = model_flavor if model_flavor else "HFtransformersV2"
    preprocess_args["vllm_enabled"] = vllm_enabled
    preprocess_args[HF_CONF.EXTRA_PIP_REQUIREMENTS.value] = extra_pip_requirements
    preprocess_args[HF_CONF.HF_CONFIG_ARGS.value] = hf_config_args
    preprocess_args[HF_CONF.HF_TOKENIZER_ARGS.value] = hf_tokenizer_args
    preprocess_args[HF_CONF.HF_MODEL_ARGS.value] = hf_model_args
    preprocess_args[HF_CONF.HF_PIPELINE_ARGS.value] = hf_pipeline_args
    preprocess_args[HF_CONF.HF_CONFIG_CLASS.value] = hf_config_class
    preprocess_args[HF_CONF.HF_PRETRAINED_CLASS.value] = hf_model_class
    preprocess_args[HF_CONF.HF_TOKENIZER_CLASS.value] = hf_tokenizer_class
    preprocess_args[HF_CONF.HF_USE_EXPERIMENTAL_FEATURES.value] = hf_use_experimental_features
    preprocess_args["inference_base_image"] = inference_base_image

    # update custom dimensions with input parameters
    custom_dimensions.update_custom_dimensions(preprocess_args)

    logger.info(f"Preprocess args : {preprocess_args}")

    with TemporaryDirectory(dir=mlflow_model_output_dir) as working_dir, TemporaryDirectory(
        dir=mlflow_model_output_dir
    ) as temp_dir:
        run_preprocess(model_framework, model_path, working_dir, temp_dir, **preprocess_args)
        shutil.copytree(working_dir, mlflow_model_output_dir, dirs_exist_ok=True)

    # Copy license file to output model path
    if license_file_path:
        # removing the default dumped license when user provides custom license file.
        transformers_license_path = os.path.join(mlflow_model_output_dir, "LICENSE.txt")
        if os.path.isfile(transformers_license_path):
            os.remove(transformers_license_path)

        shutil.copy(license_file_path, mlflow_model_output_dir)

    logger.info(f"listing output directory files: {mlflow_model_output_dir}:\n{os.listdir(mlflow_model_output_dir)}")


if __name__ == "__main__":
    run()

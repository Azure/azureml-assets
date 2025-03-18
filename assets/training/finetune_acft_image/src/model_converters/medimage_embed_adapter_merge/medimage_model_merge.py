# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for merging MedImageInsight and Adapter model."""

import json
import sys
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
import argparse
import os
import shutil


logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.model_converters.medimage_embed_adapter_merge")
COMPONENT_NAME = "ACFT-MedImage-Embedding-Classifier-ModelMerge"
TASK_TYPE = "ModelMerge"
PYTHON_MODEL = "python_model.pkl"
MLMODEL = "MLmodel"
ARTIFACTS = "artifacts/checkpoints"

MLFLOW_WRAP_CODE = "medimageinsight_classification_mlflow_wrapper.py"
CONFIG_JSON = "config.json"
CODE_FOLDER = "code"


def merge_models(mlflow_model_path, output_dir, label_file):
    """
    Merge models and prepares the output directory with necessary files and configurations.

    Args:
        mlflow_model_path (str): Path to the source MLflow model directory.
        output_dir (str): Path to the output directory where the merged model and configurations will be saved.
        label_file (str): Path to the file containing labels, one per line.
    Steps:
        1. Recursively copy contents from `mlflow_model_path` to `output_dir`.
        2. Copy the MLflow wrapper code to the output directory.
        3. Read labels from `label_file` and save them in a configuration JSON file in the output directory.
        4. Copy the MLModel configuration to the output directory.
        5. Generate a pickle model and save it to the output directory.
    Raises:
        FileNotFoundError: If any of the specified files or directories do not exist.
        IOError: If there is an error in reading or writing files.
    """
    # Copy contents from mlflow_model_path to output_dir recursively
    for root, dirs, files in os.walk(mlflow_model_path):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(output_dir, os.path.relpath(src_file, mlflow_model_path))
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy(src_file, dst_file)
            logger.info(f"Copied {src_file} to {dst_file}")

    code_dir = os.path.join(output_dir, CODE_FOLDER)
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(MLFLOW_WRAP_CODE, code_dir)
    logger.info(f"Copied {MLFLOW_WRAP_CODE} to {code_dir}")

    with open(label_file, "r") as f:
        labels = [label.strip() for label in f.read().splitlines() if label.strip()]

    config = {
        "labels": labels
    }
    with open(os.path.join(output_dir, ARTIFACTS, CONFIG_JSON), "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Copied {CONFIG_JSON} to {ARTIFACTS}")

    # Copy MLModel from configuration to mlflow_model_path+"/mlflow_model_folder/"
    shutil.copy(MLMODEL, output_dir)
    logger.info(f"Copied {MLMODEL} to {output_dir}")

    # Generate pickle model and save it to output_dir
    sys.path.insert(0, os.path.join(output_dir, CODE_FOLDER))
    import medimageinsight_classification_mlflow_wrapper
    import cloudpickle
    new_model = medimageinsight_classification_mlflow_wrapper.MEDIMAGEINSIGHTClassificationMLFlowModelWrapper(
        "image-classification",
        "medimageinsigt-v1.0.0.pt",
        "clip_tokenizer_4.16.2")
    with open(os.path.join(output_dir, PYTHON_MODEL), "wb") as f:
        cloudpickle.dump(new_model, f)
    logger.info(f"Saved {PYTHON_MODEL} to {output_dir}")


def main():
    """
    Merge an adapter model with an MLflow model.

    This function parses command-line arguments to get the paths for the MLflow model,
    the output directory to save the merged model, and an optional label file. It then
    sets the logging parameters and calls the merge_models function to perform the merge.
    Command-line Arguments:
    --mlflow_model (str): Path to the MLflow model (required).
    --output_dir (str): Directory to save the merged model (required).
    --label_file (str): Path to the label file (optional).
    Returns:
    None
    """
    parser = argparse.ArgumentParser(description="Merge adapter model with MLflow model")
    parser.add_argument("--mlflow_model", type=str, required=True, help="Path to the MLflow model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    parser.add_argument(
        '--label_file',
        type=str,
        help='Path to label file.'
    )

    args = parser.parse_args()
    set_logging_parameters(
        task_type=TASK_TYPE,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )
    merge_models(args.mlflow_model, args.output_dir, args.label_file)


if __name__ == "__main__":
    main()

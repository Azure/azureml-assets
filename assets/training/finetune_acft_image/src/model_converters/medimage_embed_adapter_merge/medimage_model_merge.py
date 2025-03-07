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
ADAPTER_MODEL = "adapter_model"
ADAPTER_CODE = "adapter_model.py"
MLFLOW_WRAP_CODE = "medimageinsight_classification_mlflow_wrapper.py"
BEST_METRIC_MODEL = "best_metric_model.pth"
CONFIG_JSON = "config.json"
CODE_FOLDER = "code"
ADAPTER_PTH = "adapter_model.pth"


def merge_models(adapter_model_path, mlflow_model_path, output_dir, hidden_dimensions, input_channels, label_file):
    logger.info(f"Merging adapter model from {adapter_model_path} with MLflow model from {mlflow_model_path}")

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
    shutil.copy(ADAPTER_CODE, code_dir)
    logger.info(f"Copied {ADAPTER_CODE} to {code_dir}")
    shutil.copy(MLFLOW_WRAP_CODE, code_dir)
    logger.info(f"Copied {MLFLOW_WRAP_CODE} to {code_dir}")

    # Copy best_metric_model.pth to mlflow_model_path+"/mlflow_model_folder/artifacts/adapter_model"
    artifacts_dir = os.path.join(output_dir, ARTIFACTS, ADAPTER_MODEL)
    os.makedirs(artifacts_dir, exist_ok=True)
    shutil.copy(os.path.join(adapter_model_path, BEST_METRIC_MODEL), os.path.join(artifacts_dir, BEST_METRIC_MODEL))
    os.rename(os.path.join(artifacts_dir, BEST_METRIC_MODEL), os.path.join(artifacts_dir, ADAPTER_PTH))
    logger.info(f"Copied {BEST_METRIC_MODEL} to {artifacts_dir}")

    with open(label_file, "r") as f:
        labels = [l.strip() for l in f.read().splitlines() if l.strip()]
        
    config = {
        "hidden_dim": hidden_dimensions,
        "in_channels": input_channels,
        "labels": labels

    }
    with open(os.path.join(artifacts_dir, CONFIG_JSON), "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Copied {CONFIG_JSON} to {artifacts_dir}")

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
    parser = argparse.ArgumentParser(description="Merge adapter model with MLflow model")
    parser.add_argument("--adapter_model", type=str, required=True, help="Path to the adapter model")
    parser.add_argument("--mlflow_model", type=str, required=True, help="Path to the MLflow model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    parser.add_argument(
        '--hidden_dimensions',
        type=int,
        required=True,
        help='Number of hidden dimensions.'
    )
    parser.add_argument(
        '--input_channels',
        type=int,
        required=True,
        help='Number of input channels.'
    )
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
    merge_models(args.adapter_model, args.mlflow_model, args.output_dir, args.hidden_dimensions, args.input_channels, args.label_file)


if __name__ == "__main__":
    main()

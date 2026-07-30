# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing functions for embeddings generation from MedImageInsight."""

import argparse
from azureml.acft.common_components import (
    get_logger_app,
    set_logging_parameters,
    LoggingLiterals,
)

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import (
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
)
import mlflow
import pandas as pd
import numpy as np
import os


COMPONENT_NAME = "ACFT-MedImage-Embedding-Generator"
EMBEDDING_FILE_NAME = "embeddings.pkl"


logger = get_logger_app(
    "azureml.acft.contrib.hf.scripts.src.process_embedding.embeddings_generator"
)
"""
Input Arguments:
    --image_tsv: Path to image TSV file.
    --mlflow_model_path: The path to the MLflow model.
    --output_pkl: Output embeddings file path.
"""


def resolve_mlflow_model_path(model_path: str) -> str:
    """Resolve the directory that contains the MLflow MLmodel file."""
    if os.path.isfile(os.path.join(model_path, "MLmodel")):
        return model_path

    for root, _, files in os.walk(model_path):
        if "MLmodel" in files:
            logger.info("Resolved MLflow model path from %s to %s", model_path, root)
            return root

    raise FileNotFoundError(f"No MLmodel file found under {model_path}")


def _read_text_preview(path: str, max_chars: int = 4000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(max_chars)
    except OSError as ex:
        return f"<failed to read {path}: {ex}>"


def _log_package_versions(package_names: list[str]) -> None:
    for package_name in package_names:
        try:
            logger.info("Package %s version: %s", package_name, importlib.metadata.version(package_name))
        except importlib.metadata.PackageNotFoundError:
            logger.info("Package %s version: <not installed>", package_name)


def log_mlflow_load_diagnostics(model_path: str) -> None:
    """Log mounted model and runtime diagnostics without dumping large files."""
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("Platform: %s", platform.platform())
    logger.info("Working directory: %s", os.getcwd())
    logger.info("MLflow version: %s", getattr(mlflow, "__version__", "<unknown>"))
    logger.info("MLflow module path: %s", getattr(mlflow, "__file__", "<unknown>"))
    logger.info("MLflow tracking URI before load: %s", mlflow.get_tracking_uri())
    logger.info("MLflow registry URI before load: %s", mlflow.get_registry_uri())
    logger.info("MLFLOW_TRACKING_URI env: %s", os.environ.get("MLFLOW_TRACKING_URI"))
    logger.info("MLFLOW_REGISTRY_URI env: %s", os.environ.get("MLFLOW_REGISTRY_URI"))
    logger.info("MLFLOW_ALLOW_FILE_STORE env: %s", os.environ.get("MLFLOW_ALLOW_FILE_STORE"))
    _log_package_versions(
        [
            "mlflow",
            "cloudpickle",
            "azure-ai-ml",
            "azureml-core",
            "azureml-dataset-runtime",
            "azureml-ai-monitoring",
            "timm",
            "transformers",
            "einops",
            "mup",
            "fvcore",
            "sentencepiece",
            "tenacity",
            "ftfy",
            "setuptools",
        ]
    )

    logger.info("Resolved model path exists: %s is_dir=%s", os.path.exists(model_path), os.path.isdir(model_path))
    for root, dirs, files in os.walk(model_path):
        rel_root = os.path.relpath(root, model_path)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth > 2:
            dirs[:] = []
            continue
        logger.info(
            "Model tree %s dirs=%s files=%s",
            rel_root,
            sorted(dirs)[:20],
            sorted(files)[:20],
        )

    for file_name in ["MLmodel", "requirements.txt", "conda.yaml", "python_env.yaml"]:
        file_path = os.path.join(model_path, file_name)
        logger.info("%s exists: %s", file_name, os.path.isfile(file_path))
        if os.path.isfile(file_path):
            logger.info("%s preview:\n%s", file_name, _read_text_preview(file_path))

    try:
        model_config = mlflow.models.Model.load(model_path)
        logger.info("MLflow Model.load flavors: %s", list(model_config.flavors.keys()))
        logger.info("MLflow Model.load metadata: %s", model_config.metadata)
    except Exception:
        logger.error("MLflow Model.load failed:\n%s", traceback.format_exc())


def load_local_mlflow_model(model_path: str):
    """Load a mounted MLflow model without AzureML tracking/registry side effects."""
    original_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    original_registry_uri = os.environ.get("MLFLOW_REGISTRY_URI")
    original_allow_file_store = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
    local_tracking_uri = "file:///tmp/mlruns"

    try:
        os.environ["MLFLOW_TRACKING_URI"] = local_tracking_uri
        os.environ["MLFLOW_REGISTRY_URI"] = local_tracking_uri
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(local_tracking_uri)
        mlflow.set_registry_uri(local_tracking_uri)
        logger.info("Loading MLflow model from %s", model_path)
        loaded_model = mlflow.pyfunc.load_model(model_path)
        logger.info("mlflow.pyfunc.load_model returned: %r", loaded_model)
        if loaded_model is not None:
            logger.info("Loaded object class: %s.%s", type(loaded_model).__module__, type(loaded_model).__name__)
            logger.info("Loaded object attributes: %s", sorted(dir(loaded_model))[:200])
        return loaded_model
    except Exception:
        logger.error("mlflow.pyfunc.load_model raised:\n%s", traceback.format_exc())
        raise
    finally:
        if original_tracking_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = original_tracking_uri
            mlflow.set_tracking_uri(original_tracking_uri)

        if original_registry_uri is None:
            os.environ.pop("MLFLOW_REGISTRY_URI", None)
        else:
            os.environ["MLFLOW_REGISTRY_URI"] = original_registry_uri
            mlflow.set_registry_uri(original_registry_uri)

        if original_allow_file_store is None:
            os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
        else:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = original_allow_file_store


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(
        description="Process medical images and get embeddigns", allow_abbrev=False
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task to be executed",
    )
    parser.add_argument("--image_tsv", type=str, help="Path to image TSV file.")
    parser.add_argument(
        "--image_standardization_jpeg_compression_ratio",
        type=int,
        default=75,
        help="JPEG compression ratio for image standardization",
    )
    parser.add_argument(
        "--image_standardization_image_size",
        type=int,
        default=512,
        help="Image size for standardization",
    )
    parser.add_argument(
        "--mlflow_model_path",
        type=str,
        required=True,
        help="The path to the MLflow model",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        help="Output PKL file path",
    )

    return parser


def generate_embeddings(

    image_tsv,
    mlflow_model,
    image_standardization_jpeg_compression_ratio,
    image_standardization_image_size,
):
    """
    Generate embeddings for images listed in a TSV file using a given MLflow model.

    Args:
        image_tsv (str): Path to the TSV file containing image data.
        mlflow_model (MLflow Model): The MLflow model used to generate image embeddings.
        image_standardization_jpeg_compression_ratio (float): JPEG compression ratio for image standardization.
        image_standardization_image_size (tuple): Target size for image standardization (width, height).
    Returns:
        pd.DataFrame: DataFrame containing the original image data along with the generated image embeddings.
    """
    image_df = pd.read_csv(image_tsv, sep="\t", header=None)
    image_df.columns = ["Name", "image"]
    image_df["text"] = None
    image_embeddings = mlflow_model.predict(
        image_df,
        params={
            "image_standardization_jpeg_compression_ratio": image_standardization_jpeg_compression_ratio,
            "image_standardization_image_size": image_standardization_image_size,
        },
    )
    image_df["features"] = image_embeddings["image_features"].apply(
        lambda item: np.array(item[0])
    )

    return image_df


def save_dataframe(
    image_embeddings: pd.DataFrame,
    output_pkl_path: str,
) -> None:
    """Save image embeddings DataFrame to an embeddings file.

    This function saves the provided image embeddings DataFrame
    to the specified output path with the given file name. It also creates
    the directory if it does not exist.

    Args:
        image_embeddings (pd.DataFrame): The DataFrame containing image embeddings to be saved.
        output_pkl_path (str): The directory path where the embeddings file will be saved.
    Returns:
        None
    """
    os.makedirs(output_pkl_path, exist_ok=True)

    image_embeddings.to_json(
        os.path.join(output_pkl_path, EMBEDDING_FILE_NAME),
        orient="records",
        lines=True,
    )

    logger.info("Saved merged DataFrames to embeddings file")


def process_embeddings(args):
    """
    Process medical image embeddings and save the results to a PKL file.

    This function loads the MLflow model, generates image embeddings from the provided TSV file,
    and saves the embeddings to the specified output PKL file.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - mlflow_model_path (str): The path to the MLflow model.
            - image_tsv (str): The path to the image TSV file.
            - output_pkl (str): The path to save the output embeddings file.
    Returns:
        None
    """
    model_path = args.mlflow_model_path
    output_pkl = args.output_pkl
    image_tsv = args.image_tsv

    resolved_model_path = resolve_mlflow_model_path(model_path)
    log_mlflow_load_diagnostics(resolved_model_path)
    mlflow_model = load_local_mlflow_model(resolved_model_path)
    if mlflow_model is None:
        raise RuntimeError(f"mlflow.pyfunc.load_model returned None for {resolved_model_path}")
    logger.info("Loaded MLflow model type: %s", type(mlflow_model).__name__)
    image_embeddings = generate_embeddings(
        image_tsv,
        mlflow_model,
        args.image_standardization_jpeg_compression_ratio,
        args.image_standardization_image_size,
    )

    save_dataframe(image_embeddings, output_pkl)

    logger.info("Processing medical images and getting embeddings completed")


def main():
    """
    To parse arguments, set logging parameters, and process embeddings.

    This function performs the following steps:
    1. Parses command-line arguments using a parser.
    2. Logs the parsed arguments.
    3. Sets logging parameters including task type, project name, project version number, and component name.
    4. Filters specific logging patterns for Azure Machine Learning.
    5. Processes embeddings based on the parsed arguments.
    Returns:
        None
    """
    parser = get_parser()
    args, _ = parser.parse_known_args()
    logger.info("Parsed arguments: %s", args)

    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )
    logger.info("Logging parameters set")

    process_embeddings(args)


if __name__ == "__main__":
    main()

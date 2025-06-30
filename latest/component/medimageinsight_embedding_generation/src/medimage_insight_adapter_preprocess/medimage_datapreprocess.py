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
    --output_pkl: Output PKL file path.
"""


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
    """Save image embeddings DataFrame to a PKL file.

    This function saves the provided image embeddings DataFrame
    to the specified PKL file path with the given file name. It also creates
    the directory if it does not exist.

    Args:
        image_embeddings (pd.DataFrame): The DataFrame containing image embeddings to be saved.
        output_pkl_path (str): The directory path where the PKL file will be saved.
    Returns:
        None
    """
    os.makedirs(output_pkl_path, exist_ok=True)

    image_embeddings.to_pickle(os.path.join(output_pkl_path, EMBEDDING_FILE_NAME))

    logger.info("Saved merged DataFrames to PKL files")


def process_embeddings(args):
    """
    Process medical image embeddings and save the results to a PKL file.

    This function loads the MLflow model, generates image embeddings from the provided TSV file,
    and saves the embeddings to the specified output PKL file.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - mlflow_model_path (str): The path to the MLflow model.
            - image_tsv (str): The path to the image TSV file.
            - output_pkl (str): The path to save the output PKL file.
    Returns:
        None
    """
    model_path = args.mlflow_model_path
    output_pkl = args.output_pkl
    image_tsv = args.image_tsv

    mlflow_model = mlflow.pyfunc.load_model(model_path)
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

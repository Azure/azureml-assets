import argparse
from azureml.acft.common_components import (
    get_logger_app,
    set_logging_parameters,
    LoggingLiterals,
)
from azureml.acft.common_components.utils.error_handling.exceptions import (
    ACFTValidationException,
)
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    ACFTUserError,
)
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml._common._error_definition.azureml_error import AzureMLError

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import (
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
)
import mlflow
import pandas as pd
import numpy as np
import os
import json


COMPONENT_NAME = "ACFT-MedImage-Embedding-Generator"
TRAIN_EMBEDDING_FILE_NAME = "train_embeddings.pkl"
VALIDATION_EMBEDDING_FILE_NAME = "validation_embeddings.pkl"


logger = get_logger_app(
    "azureml.acft.contrib.hf.scripts.src.process_embedding.embeddings_generator"
)
"""
Input Arguments: endpoint_url, endpoint_key, zeroshot_path, test_train_split_pkl_path
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
        "--eval_image_tsv", type=str, help="Path to evaluation image TSV file."
    )
    parser.add_argument(
        "--eval_text_tsv", type=str, help="Path to evaluation text TSV file."
    )
    parser.add_argument(
        "--image_tsv", type=str, help="Path to training image TSV file."
    )
    parser.add_argument("--text_tsv", type=str, help="Path to training text TSV file.")
    parser.add_argument(
        "--mlflow_model_path",
        type=str,
        required=True,
        help="The path to the MLflow model",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task to be executed",
    )
    parser.add_argument(
        "--output_train_pkl",
        type=str,
        help="Output train PKL file path",
    )
    parser.add_argument(
        "--output_validation_pkl",
        type=str,
        help="Output validation PKL file path",
    )

    return parser


def generate_embeddings(image_tsv, text_tsv, mlflow_model):
    image_df = pd.read_csv(image_tsv, sep="\t")
    image_df.columns = ["Name", "image"]
    image_df["text"] = None
    image_embeddings = mlflow_model.predict(image_df)
    image_df["features"] = image_embeddings["image_features"].apply(lambda item: np.array(item[0]))

    text_df = pd.read_csv(text_tsv, sep="\t")
    text_df.columns = ["Name", "classification"]

    def extract_text_field(text):
        try:
            text_json = json.loads(text)
            return text_json.get("class_id", -1)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from text column")
            return ""

    text_df["Label"] = text_df["classification"].apply(extract_text_field)
    return pd.merge(image_df, text_df, on="Name", how="inner")


def save_merged_dataframes(
    train_merged: pd.DataFrame,
    val_merged: pd.DataFrame,
    output_train_pkl_path: str,
    output_validation_pkl_path: str,
    train_embedding_file_name: str,
    validation_embedding_file_name: str,
) -> None:
    """Save merged DataFrames to PKL files.

    This function saves the provided training and validation merged DataFrames
    to the specified PKL file paths with the given file names. It also creates
    the directories if they do not exist.

    Args:
        train_merged (pd.DataFrame): The merged training DataFrame to be saved.
        val_merged (pd.DataFrame): The merged validation DataFrame to be saved.
        output_train_pkl_path (str): The directory path where the training PKL file will be saved.
        output_validation_pkl_path (str): The directory path where the validation PKL file will be saved.
        train_embedding_file_name (str): The file name for the training PKL file.
        validation_embedding_file_name (str): The file name for the validation PKL file.
    Returns:
        None
    """
    os.makedirs(output_train_pkl_path, exist_ok=True)
    os.makedirs(output_validation_pkl_path, exist_ok=True)

    train_merged.to_pickle(
        os.path.join(output_train_pkl_path, train_embedding_file_name)
    )
    val_merged.to_pickle(
        os.path.join(output_validation_pkl_path, validation_embedding_file_name)
    )
    logger.info("Saved merged DataFrames to PKL files")


def process_embeddings(args):
    """
    Process medical image embeddings and save the results to PKL files.
    This function initializes the medimageinsight object, generates image embeddings,
    creates a features dataframe, loads train and validation PKL files, merges the dataframes,
    and saves the merged dataframes to specified output PKL files.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - mlflow_model_path (str): The path to the MLflow model.
            - zeroshot_path (str): The path to the zeroshot data.
            - output_train_pkl (str): The path to save the output training PKL file.
            - output_validation_pkl (str): The path to save the output validation PKL file.
            - test_train_split_csv_path (str): The path to the test/train split CSV file.
    Returns:
        None
    """

    model_path = args.mlflow_model_path
    output_train_pkl = args.output_train_pkl
    output_validation_pkl = args.output_validation_pkl
    image_tsv = args.image_tsv
    text_tsv = args.text_tsv
    eval_image_tsv = args.eval_image_tsv
    eval_text_tsv = args.eval_text_tsv

    mlflow_model = mlflow.pyfunc.load_model(model_path)
    image_embeddings = generate_embeddings(image_tsv, text_tsv, mlflow_model)
    eval_image_embeddings = generate_embeddings(
        eval_image_tsv, eval_text_tsv, mlflow_model
    )

    save_merged_dataframes(
        image_embeddings,
        eval_image_embeddings,
        output_train_pkl,
        output_validation_pkl,
        TRAIN_EMBEDDING_FILE_NAME,
        VALIDATION_EMBEDDING_FILE_NAME,
    )

    logger.info("Processing medical images and getting embeddings completed")


def main():
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

"""
python medimage_datapreprocess.py --task_name "MedEmbedding" --mlflow_model_path "/mnt/model/MedImageInsight/mlflow_model_folder" --zeroshot_path "/home/healthcare-ai/medimageinsight-zeroshot/" --test_train_split_csv_path "/home/healthcare-ai/medimageinsight/classification_demo/data_input/" --output_train_pkl "/home/healthcare-ai/" --output_validation_pkl "/home/healthcare-ai/"

"""

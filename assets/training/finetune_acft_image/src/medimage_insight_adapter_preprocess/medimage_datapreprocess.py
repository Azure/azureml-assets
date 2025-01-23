import argparse
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml._common._error_definition.azureml_error import AzureMLError

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
import pandas as pd
import torch
import os
from classification_demo.MedImageInsight import medimageinsight_package
from classification_demo.adaptor_training import training
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# Suppress SimpleITK warnings
sitk.ProcessObject_SetGlobalWarningDisplay(False)


COMPONENT_NAME = "ACFT-MedImage-Embedding-Generator"
TRAIN_EMBEDDING_FILE_NAME = "train_embeddings.pkl"
VALIDATION_EMBEDDING_FILE_NAME = "validation_embeddings.pkl"


logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.process_embedding.embeddings_generator")
'''
Input Arguments: endpoint_url, endpoint_key, zeroshot_path, test_train_split_pkl_path
'''


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description='Process medical images and get embeddigns', allow_abbrev=False)

    parser.add_argument(
        '--endpoint_url',
        type=str,
        required=True,
        help='The endpoint URL embedding generation model.'
        )
    parser.add_argument(
        '--endpoint_key',
        type=str,
        required=True,
        help='The endpoint key for the embedding generation model.'
        )
    parser.add_argument(
        '--zeroshot_path',
        type=str,
        required=True,
        help='The path to the zeroshot dataset'
        )
    parser.add_argument(
        '--test_train_split_csv_path',
        type=str,
        required=True,
        help='The path to the test/train split CSV file'
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


def load_csv_files(test_train_split_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation CSV files into DataFrames.

    This function loads the training and validation csv files from the specified path
    and returns them as pandas DataFrames.

    Args:
        test_train_split_csv_path (str): The path to the directory containing the test/train split CSV files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation DataFrames.
    """
    train_csv_path = f"{test_train_split_csv_path}/adaptor_tutorial_train_split.csv"
    val_csv_path = f"{test_train_split_csv_path}/adaptor_tutorial_test_split.csv"
    logger.info("Train CSV path: %s", train_csv_path)
    logger.info("Validation CSV path: %s", val_csv_path)

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    logger.info("Loaded training and validation CSV files into DataFrames")
    return train_df, val_df


def create_features_dataframe(image_embedding_dict: dict) -> pd.DataFrame:
    """
    Create a DataFrame from image embeddings.

    This function creates a DataFrame from the provided image embedding dictionary.
    The DataFrame contains two columns: "Name" and "features".

    Args:
        image_embedding_dict (dict): A dictionary containing image embeddings.

    Returns:
        pd.DataFrame: A DataFrame containing the image features.
    """
    df_features = pd.DataFrame(
        {
            "Name": list(image_embedding_dict.keys()),
            "features": [v["image_feature"] for v in image_embedding_dict.values()],
        }
    )
    logger.info("Created DataFrame for image features")
    return df_features


def generate_image_embeddings(medimageinsight: medimageinsight_package, zeroshot_path: str) -> dict:
    """
    Generate image embeddings using the MedImageInsight package.

    This function generates image embeddings for the provided zeroshot path using the MedImageInsight package.

    Args:
        medimageinsight (medimageinsight_package): An instance of the MedImageInsight package.
        zeroshot_path (str): The path to the zeroshot data.

    Returns:
        dict: A dictionary containing the image embeddings.
    """
    image_embedding_dict, _ = medimageinsight.generate_embeddings(
        data={"image": zeroshot_path, "text": None, "params": {"get_scaling_factor": False}}
    )
    logger.info("Generated embeddings for images")
    return image_embedding_dict


def initialize_medimageinsight(endpoint_url: str, endpoint_key: str) -> medimageinsight_package:
    """
    Initialize the MedImageInsight package.

    This function initializes the MedImageInsight package using the provided endpoint URL and key.

    Args:
        endpoint_url (str): The URL of the endpoint.
        endpoint_key (str): The key for the endpoint.

    Returns:
        medimageinsight_package: An instance of the MedImageInsight package.
    """
    medimageinsight = medimageinsight_package(
        endpoint_url=endpoint_url,
        endpoint_key=endpoint_key,
    )
    logger.info("Initialized MedImageInsight package")
    return medimageinsight


def merge_dataframes(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge training and validation DataFrames with features DataFrame.

    This function merges the provided training and validation DataFrames with the features DataFrame
    based on the "Name" column.

    Args:
        train_df (pd.DataFrame): The training DataFrame.
        val_df (pd.DataFrame): The validation DataFrame.
        df_features (pd.DataFrame): The features DataFrame containing image features.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the merged training and validation DataFrames.
    """
    train_merged = pd.merge(train_df, df_features, on="Name", how="inner")
    val_merged = pd.merge(val_df, df_features, on="Name", how="inner")
    logger.info("Merged training and validation DataFrames with features DataFrame")
    return train_merged, val_merged


def save_merged_dataframes(train_merged: pd.DataFrame, val_merged: pd.DataFrame, output_train_pkl_path: str,
                           output_validation_pkl_path: str, train_embedding_file_name: str,
                           validation_embedding_file_name: str) -> None:
    """ Save merged DataFrames to PKL files.

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

    train_merged.to_pickle(os.path.join(output_train_pkl_path, train_embedding_file_name))
    val_merged.to_pickle(os.path.join(output_validation_pkl_path, validation_embedding_file_name))
    logger.info("Saved merged DataFrames to PKL files")


def process_embeddings(args):
    """
    Process medical image embeddings and save the results to PKL files.
    This function initializes the medimageinsight object, generates image embeddings,
    creates a features dataframe, loads train and validation PKL files, merges the dataframes,
    and saves the merged dataframes to specified output PKL files.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - endpoint_url (str): The URL of the endpoint.
            - endpoint_key (str): The key for the endpoint.
            - zeroshot_path (str): The path to the zeroshot data.
            - output_train_pkl (str): The path to save the output training PKL file.
            - output_validation_pkl (str): The path to save the output validation PKL file.
            - test_train_split_csv_path (str): The path to the test/train split CSV file.
    Returns:
        None
    """
    endpoint_url = args.endpoint_url
    endpoint_key = args.endpoint_key
    zeroshot_path = args.zeroshot_path
    output_train_pkl = args.output_train_pkl
    output_validation_pkl = args.output_validation_pkl
    test_train_split_csv_path = args.test_train_split_csv_path

    logger.info("Endpoint URL: %s", endpoint_url)
    logger.info("Zeroshot path: %s", zeroshot_path)
    logger.info("Test/train split PKL path: %s", test_train_split_csv_path)

    medimageinsight = initialize_medimageinsight(endpoint_url, endpoint_key)
    image_embedding_dict = generate_image_embeddings(medimageinsight, zeroshot_path)
    df_features = create_features_dataframe(image_embedding_dict)

    train_df, val_df = load_csv_files(test_train_split_csv_path)
    train_merged, val_merged = merge_dataframes(train_df, val_df, df_features)

    save_merged_dataframes(train_merged, val_merged, output_train_pkl, output_validation_pkl,
                           TRAIN_EMBEDDING_FILE_NAME, VALIDATION_EMBEDDING_FILE_NAME)
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
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )
    logger.info("Logging parameters set")

    process_embeddings(args)


if __name__ == '__main__':
    main()

'''
python medimage_datapreprocess.py --task_name "MedEmbedding" --endpoint_url "https://medimageinsight-od3wv.eastus.inference.ml.azure.com/score" --endpoint_key "DFe8Z8VkwDFgdNe8kQJOWBqglpdOwxBn" --zeroshot_path "/home/healthcare-ai/medimageinsight-zeroshot/" --test_train_split_csv_path "/home/healthcare-ai/medimageinsight/classification_demo/data_input/" --output_train_pkl "/home/healthcare-ai/" --output_validation_pkl "/home/healthcare-ai/"

'''

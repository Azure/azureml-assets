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


COMPONENT_NAME = "ACFT-MedImage-Classification-Training"
logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.train.classification_adaptor_train")
TRAIN_EMBEDDING_FILE_NAME = "train_embeddings.pkl"
VALIDATION_EMBEDDING_FILE_NAME = "validation_embeddings.pkl"


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description='Process medical images and get embeddings', allow_abbrev=False)

    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task to be executed",
    )
    parser.add_argument(
        '--train_data_path',
        type=str,
        required=True,
        help='The path to the training data.'
    )
    parser.add_argument(
        '--validation_data_path',
        type=str,
        required=True,
        help='The path to the validation data.'
    )
    parser.add_argument(
        '--train_dataloader_batch_size',
        type=int,
        required=True,
        help='Batch size for the training dataloader.'
    )
    parser.add_argument(
        '--validation_dataloader_batch_size',
        type=int,
        required=True,
        help='Batch size for the validation dataloader.'
    )
    parser.add_argument(
        '--train_dataloader_workers',
        type=int,
        required=True,
        help='Number of workers for the training dataloader.'
    )
    parser.add_argument(
        '--validation_dataloader_workers',
        type=int,
        required=True,
        help='Number of workers for the validation dataloader.'
    )
    parser.add_argument(
        '--output_classes',
        type=int,
        required=True,
        help='Number of output classes.'
    )
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
        '--learning_rate',
        type=float,
        required=True,
        help='Learning rate for the model.'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        required=True,
        help='Maximum number of epochs for training.'
    )
    parser.add_argument(
        '--output_model_path',
        type=str,
        required=True,
        help='Path to save the output model.'
    )

    return parser


def load_data(train_data_path: str, validation_data_path: str, train_file_name: str,
              validation_file_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and validation data from the provided folder paths.

    Args:
        train_data_path (str): The path to the folder containing the training data file.
        validation_data_path (str): The path to the folder containing the validation data file.
        train_file_name (str): The name of the training data file.
        validation_file_name (str): The name of the validation data file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the training and validation data.
    """

    train_data_file = os.path.join(train_data_path, train_file_name)
    validation_data_file = os.path.join(validation_data_path, validation_file_name)
    train_data = pd.read_pickle(train_data_file)
    validation_data = pd.read_pickle(validation_data_file)
    return train_data, validation_data


def initialize_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    Initialize the model with the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        torch.nn.Module: Initialized model.
    """
    return training.create_model(
        in_channels=args.input_channels,
        hidden_dim=args.hidden_dimensions,
        num_class=args.output_classes
    )


def prepare_dataloaders(
    train_data: pd.DataFrame, 
    validation_data: pd.DataFrame, 
    args: argparse.Namespace
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepare the dataloaders for training and validation datasets.

    Args:
        train_data (pd.DataFrame): DataFrame containing the training data.
        validation_data (pd.DataFrame): DataFrame containing the validation data.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
            Dataloaders for the training and validation datasets.
    """
    train_samples = {
        "features": train_data["features"].tolist(),
        "img_name": train_data["Name"].tolist(),
        "labels": train_data["Label"].tolist(),
    }
    val_samples = {
        "features": validation_data["features"].tolist(),
        "img_name": validation_data["Name"].tolist(),
        "labels": validation_data["Label"].tolist(),
    }
    train_dataloader = training.create_data_loader(
        train_samples,
        csv=train_data,
        mode="train",
        batch_size=args.train_dataloader_batch_size,
        num_workers=args.train_dataloader_workers,
        pin_memory=True
    )
    validation_dataloader = training.create_data_loader(
        val_samples,
        csv=validation_data,
        mode="val",
        batch_size=args.validation_dataloader_batch_size,
        num_workers=args.validation_dataloader_workers,
        pin_memory=True
    )
    return train_dataloader, validation_dataloader


def train_model(
    train_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    args: argparse.Namespace
) -> tuple[float, float]:
    """
    Train the model using the provided dataloaders and arguments.

    Args:
        train_dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        validation_dataloader (torch.utils.data.DataLoader): Dataloader for the validation data.
        model (torch.nn.Module): The model to be trained.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple[float, float]: Best accuracy and best AUC achieved during training.
    """
    learning_rate = args.learning_rate
    loss_function_ts = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir = args.output_model_path
    training.create_output_directory(output_dir)

    best_accuracy, best_auc = training.trainer(
        train_dataloader,
        validation_dataloader,
        model,
        loss_function_ts,
        optimizer,
        epochs=int(args.max_epochs),
        root_dir=output_dir,
    )
    return best_accuracy, best_auc


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
    train_data, validation_data = load_data(args.train_data_path, args.validation_data_path,
                                            TRAIN_EMBEDDING_FILE_NAME, VALIDATION_EMBEDDING_FILE_NAME)
    model = initialize_model(args)
    train_dataloader, validation_dataloader = prepare_dataloaders(train_data, validation_data, args)
    best_accuracy, best_auc = train_model(train_dataloader, validation_dataloader, model, args)
    print(f"Best Accuracy of the Adaptor: {best_accuracy:.4f}")
    print(f"Best AUC of the Adaptor: {best_auc:.4f}")


if __name__ == "__main__":
    main()

# Example command to run this script:
# python medimage_train.py --task_name "AdapterTrain" --train_data_path "/home/healthcare-ai/train_merged.pkl" --validation_data_path "/home/healthcare-ai/val_merged.pkl" --train_dataloader_batch_size 8 --validation_dataloader_batch_size 1 --train_dataloader_workers 2 --validation_dataloader_workers 2 --output_classes 5 --hidden_dimensions 512 --input_channels 1024 --learning_rate 0.0003 --max_epochs 10 --output_model_path "/home/healthcare-ai/"

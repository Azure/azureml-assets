# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Dataset Downloader Component."""

import argparse
import os
import tempfile
from typing import Optional, List

from datasets import Features, get_dataset_split_names, get_dataset_config_names, load_dataset, Value
import pandas as pd
from azureml._common._error_definition.azureml_error import AzureMLError

from aml_benchmark.dataset_downloader.vision_dataset_adapter import VisionDatasetAdapterFactory
from aml_benchmark.utils.aml_run_utils import get_default_datastore
from aml_benchmark.utils.logging import get_logger, log_mlflow_params
from aml_benchmark.utils.io import get_image_file_name, get_random_datastore_directory
from aml_benchmark.utils.exceptions import (
    swallow_all_exceptions,
    BenchmarkValidationException,
    DatasetDownloadException,
)
from aml_benchmark.utils.error_definitions import BenchmarkValidationError, DatasetDownloadError


logger = get_logger(__name__)

ALL = "all"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to download.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default=None,
        help=(
            "Sub-part of the dataset to download; specify 'all' to download all sub-parts; specify "
            "comma-separated values to download multiple sub-parts (Ex: config1,config2)."
        )
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split of the dataset to download; specify 'all' to download all splits.",
    )
    parser.add_argument(
        "--script",
        default=None,
        type=str,
        help=(
            "Path to the dataset loading script. Must follow the HuggingFace dataset loading script template. "
            "For example, please refer "
            "https://github.com/Azure/azureml-assets/tree/main/assets/aml-benchmark/scripts/data_loaders."
        )
    )
    parser.add_argument(
        "--output_dataset", type=str, required=True, help="Path to the directory where the dataset will be downloaded."
    )

    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def resolve_configuration(
    dataset_name: Optional[str], configuration: Optional[str]
) -> List[Optional[str]]:
    """
    Get the list of configurations to download.

    :param dataset_name: Name of the dataset to download.
    :param configuration: Configuration of the dataset to download.
    :return: list of configurations to download.
    """
    available_configs = get_dataset_config_names(dataset_name)

    if configuration is None:
        if len(available_configs) > 1:
            mssg = (
                f"Multiple configurations available for dataset '{dataset_name}'. Please specify either one of "
                f"the following: {available_configs} or 'all'."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        else:
            return [configuration]

    if configuration == ALL:
        return available_configs

    configuration_arr = configuration.split(",")
    for configuration in configuration_arr:
        if configuration not in available_configs:
            mssg = f"Configuration '{configuration}' not available for dataset '{dataset_name}'."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
    return configuration_arr


def resolve_split(
    dataset_name: str, configuration: Optional[str], split: str
) -> List[str]:
    """
    Get the list of splits to download.

    :param dataset_name: Name of the dataset to download.
    :param configuration: Configuration of the dataset to download.
    :param split: Split of the dataset to download.
    :return: list of splits to download.
    """
    available_splits = get_dataset_split_names(
        path=dataset_name, config_name=configuration
    )
    if split == ALL:
        return available_splits

    if split not in available_splits:
        mssg = f"Split '{split}' not available for dataset '{dataset_name}' and config '{configuration}'."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )
    return [split]


def download_dataset_from_hf(
    dataset_name: str, configuration: Optional[str], split: str, output_dir: str
) -> None:
    """
    Download a dataset from HuggingFace and save it in the supplied directory.

    :param dataset_name: Name of the dataset to download.
    :param configuration: Configuration of the dataset to download.
    :param split: Split of the dataset to download.
    :param output_dir: Directory path where the dataset will be downloaded.
    :return: None
    """
    splits = resolve_split(dataset_name, configuration, split)
    for split in splits:
        logger.info(f"Downloading - Configuration: '{configuration}', Split: '{split}'.")
        try:
            dataset = load_dataset(path=dataset_name, name=configuration, split=split)
        except Exception as e:
            mssg = f"Error downloading dataset: {e}"
            raise DatasetDownloadException._with_error(
                AzureMLError.create(DatasetDownloadError, error_details=mssg)
            )
        out_dir = os.path.join(output_dir, str(configuration), split)
        os.makedirs(out_dir, exist_ok=True)
        output_file_path = os.path.join(out_dir, "data.jsonl")

        # Save dataset to JSONL format, making the necessary adaptations for vision (eg save image data to datastore
        # and include image links in JSONL file).
        vision_dataset_adapter = VisionDatasetAdapterFactory.get_adapter(dataset)
        _save_dataset_to_jsonl(dataset, vision_dataset_adapter, output_file_path)

        logger.info(f"Downloaded - Configuration: '{configuration}', Split: '{split}'.")


def _save_dataset_to_jsonl(dataset, vision_dataset_adapter, dataset_file_path):
    logger.info("a1 {} {}".format(vision_dataset_adapter, dataset_file_path))

    # Local temporary directory not used by default.
    temporary_directory = None

    if vision_dataset_adapter is not None:
        # Make the local temporary directory where the images will be saved.
        temporary_directory = tempfile.TemporaryDirectory()

        # Get the datastore and the directory within it where the images will be uploaded.
        datastore = get_default_datastore()
        logger.info("a2 {}".format(datastore))
        datastore_directory_name, datastore_directory_url = get_random_datastore_directory(datastore.name)
        logger.info("a3 {} {}".format(datastore_directory_name, datastore_directory_url))

        # Initialize the image counter.
        image_counter = 0

        def adapt_vision_instance(instance):
            nonlocal image_counter

            if image_counter < 2:
                logger.info("a4 {}".format(list(instance.keys())))

            # Save the current image to the local temporary directory.
            image_file_name = get_image_file_name(image_counter=image_counter)
            image_file_path = os.path.join(temporary_directory.name, image_file_name)
            vision_dataset_adapter.get_pil_image(instance).save(image_file_path)

            if image_counter < 2:
                logger.info("a5 {} {}".format(type(vision_dataset_adapter.get_pil_image(instance)), image_file_path))

            # Get the current label.
            label = vision_dataset_adapter.get_label(instance)

            if image_counter < 2:
                logger.info("a6 {} {}".format(type(label), label))

            # Since returning a different dictionary than `instance` does not work, the current dictionary is modified
            # and then returned.
            instance.clear()

            # Store the datastore URL where the image will be saved and also store the label.
            instance["image_url"] = "{}/{}".format(datastore_directory_url, image_file_name)
            instance["label"] = label

            if image_counter < 2:
                logger.info("a7 {}".format(instance))

            # Increment the image counter.
            image_counter += 1

            return instance

        logger.info("a8")

        # Save the images in the dataset to temporary local storage.
        dataset = dataset.map(
            adapt_vision_instance,
            features=Features({
                "image_url": Value(dtype="string", id=None),
                "label": Value(dtype="string", id=None),
            })
        )
        logger.info("a9", dataset[0])

        # Upload the images to datastore.
        datastore.upload(src_dir=temporary_directory.name, target_path=datastore_directory_name)

    logger.info("a10")

    # Save the dataset to JSONL format.
    dataset.to_json(dataset_file_path)

    logger.info("a11")

    # If images were saved to local temporary directory, delete it.
    if temporary_directory is not None:
        logger.info("a12")
        temporary_directory.cleanup()


def download_file_from_url(url: str, output_dir: str) -> None:
    """
    Download a file from a URL and saves it in the supplied file path.

    :param url: URL of the file to download.
    :param output_dir: Path of the directory where the dataset will be downloaded.
    :return: None
    """
    file_name = url.split("/")[-1]
    file_ext = file_name.split(".")[-1]
    try:
        if file_ext == "csv":
            df = pd.read_csv(url)
        elif file_ext == "tsv":
            df = pd.read_csv(url, sep="\t")
        elif file_ext == "txt":
            df = pd.read_csv(url, sep=" ")
        elif file_ext == "pkl":
            df = pd.read_pickle(url)
        elif file_ext == "xls" or file_ext == "xlsx":
            df = pd.read_excel(url)
        elif file_ext == "json":
            df = pd.read_json(url)
        elif file_ext == "jsonl":
            df = pd.read_json(url, lines=True)
        elif file_ext == "parquet":
            df = pd.read_parquet(url)
        elif file_ext == "feather":
            df = pd.read_feather(url)
        elif file_ext == "hdf":
            df = pd.read_hdf(url)
        else:
            mssg = f"File extension '{file_ext}' not supported."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
    except Exception as e:
        mssg = f"Error downloading file: {e}"
        raise DatasetDownloadException._with_error(
            AzureMLError.create(DatasetDownloadError, error_details=mssg)
        )

    output_file_path = os.path.join(output_dir, "data.jsonl")
    df.to_json(output_file_path, lines=True, orient="records")
    logger.info(f"File downloaded at '{output_file_path}'.")


@swallow_all_exceptions(logger)
def main(
    output_dataset: str,
    split: str,
    dataset_name: Optional[str] = None,
    configuration: Optional[str] = None,
    script: Optional[str] = None,
) -> None:
    """
    Entry function for Dataset Downloader.

    Either `dataset_name` or `script` must be supplied; but not both.

    :param output_dataset: Path to the directory where the dataset will be downloaded.
    :param split: Split of the dataset to download.
    :param dataset_name: Name of the dataset to download.
    :param configuration: Sub-dataset of the dataset to download.
    :param script: Path to the dataset loading script. Must follow the HuggingFace dataset loading script template.
    :return: None
    """
    # Check if dataset_name and script are supplied
    if dataset_name and script:
        mssg = "Either 'dataset_name' or 'script' must be supplied; but not both."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    # Check if dataset_name or script is supplied
    if not (dataset_name or script):
        mssg = "Either 'dataset_name' or 'script' must be supplied."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    if not dataset_name:
        dataset_name = script

    # Get the configurations to download
    try:
        configurations = resolve_configuration(dataset_name, configuration)
    except FileNotFoundError as e:
        mssg = f"FileNotFoundError: {e}"
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    logger.info(f"Following configurations will be downloaded: {configurations}.")
    for config in configurations:
        download_dataset_from_hf(
            dataset_name,
            config,
            split,
            output_dataset,
        )

    log_mlflow_params(
        dataset_name=dataset_name if script is None else None,
        configuration=configuration,
        split=split,
        script=script,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_name=args.dataset_name,
        configuration=args.configuration,
        split=args.split,
        script=args.script,
        output_dataset=args.output_dataset,
    )

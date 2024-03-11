# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""data upload component."""
from typing import Any, Dict
import argparse
from io import BytesIO
from openai import AzureOpenAI
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.utils import save_json
from common.logging import get_logger
import jsonlines
import os

logger = get_logger(__name__)


class UploadComponent:
    """ Upload component to upload data to AOAI."""

    def __init__(self, aoai_client: AzureOpenAI):
        """ Upload component to upload data to AOAI."""
        self.aoai_client = aoai_client
        self.split_ratio = 0.8

    def upload_files(self, train_file_path: str, validation_file_path: str = None):
        """ Uploads training and validation files to azure openai """

        train_file_name = os.path.basename(train_file_path)
        validation_file_name = os.path.basename(validation_file_path)\
            if validation_file_path is not None else "validation_" + train_file_name

        if validation_file_path is not None:
            train_data = open(train_file_path, "rb")
            validation_data = open(validation_file_path, "rb")
        else:
            logger.debug(f"validation data not provided,\
                        splitting train data in ratio : {self.split_ratio} to create validation data")
            train_data, validation_data = self._get_train_validation_split_data(train_file_path)

        logger.debug(f"uploading training file : {train_file_name} and validation file : {validation_file_name}")
        train_metadata = self._upload_file(train_file_name, train_data)
        validation_metadata = self._upload_file(validation_file_name, validation_data)
        upload_files_output: Dict[str, Dict[str, Any]] = {"train_file_id": train_metadata.id,
                                                          "validation_file_id": validation_metadata.id}
        return upload_files_output

    def _upload_file(self, file_name, file_data):
        upload_file_metadata = self.aoai_client.files.create(file=(file_name, file_data, 'application/json'),
                                                             purpose='fine-tune')
        upload_file_metadata = self.aoai_client.files.wait_for_processing(upload_file_metadata.id)
        self._check_upload_status(upload_file_metadata.dict())
        return upload_file_metadata

    def _get_train_validation_split_data(self, train_file_path):
        split_index = self._get_split_index(train_file_path)
        index = 0
        train_data = BytesIO()
        validation_data = BytesIO()

        with open(train_file_path, "rb") as train_data_reader:
            for line in train_data_reader:
                if index < split_index:
                    train_data.write(line)
                else:
                    validation_data.write(line)
                index += 1
        return train_data, validation_data

    def _check_upload_status(self, upload_file_metadata_dictionary: dict):
        """ Check upload status."""
        file_id = upload_file_metadata_dictionary["id"]
        filename = upload_file_metadata_dictionary["filename"]
        upload_file_status = upload_file_metadata_dictionary["status"]

        if upload_file_status == "error":
            error_reason = upload_file_metadata_dictionary["error"]
            error_string = f"uploading file failed for {filename}, file id: {file_id}, reason: {error_reason}"
            logger.error(error_string)
            raise Exception(error_string)

        logger.debug("file: {}, file id: {} uploaded with status: {}".format(filename, file_id, upload_file_status))

    def _get_split_index(self, train_file_path):
        with jsonlines.open(train_file_path, mode='r') as file_reader:
            train_data_len = sum(1 for _ in file_reader)
        split_index = int(train_data_len * self.split_ratio)
        logger.info(f"total length of train data: {train_data_len}, split_index: {split_index}")
        return split_index


def main():
    """ Main function to upload data to AOAI."""

    parser = argparse.ArgumentParser(description="Upload Component")
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--validation_dataset", type=str)
    parser.add_argument("--dataset_upload_output", type=str)
    parser.add_argument("--endpoint_name", type=str)
    parser.add_argument("--endpoint_resource_group", type=str)
    parser.add_argument("--endpoint_subscription", type=str)

    args = parser.parse_args()

    try:
        # aoai_client = aoai_utils.get_azure_oai_client()
        aoai_client_manager = AzureOpenAIClientManager(endpoint_name=args.endpoint_name,
                                                       endpoint_resource_group=args.endpoint_resource_group,
                                                       endpoint_subscription=args.endpoint_subscription)
        upload_component = UploadComponent(aoai_client_manager.get_azure_openai_client())

        dataset_upload_output = upload_component.upload_files(args.train_dataset, args.validation_dataset)
        save_json(dataset_upload_output, args.dataset_upload_output)
        logger.info("uploaded train and validation data")

    except Exception as e:
        logger.error("Got exception while running Upload data component. Ex: {}".format(e))
        raise e


if __name__ == "__main__":

    main()

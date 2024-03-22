# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""data delete component."""
from typing import Any, Dict
import argparse
from openai import AzureOpenAI
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.utils import save_json
from common.logging import get_logger
import json
import os

logger = get_logger(__name__)


class DeleteComponent:
    """Delete component to delete data from AOAI resource."""

    def __init__(self, aoai_client: AzureOpenAI):
        """Delete component to delete data from AOAI resource."""
        self.aoai_client = aoai_client

    def delete_files(self, data_upload_output):
        """Delete component to delete data from AOAI resource."""

        with open(data_upload_output) as f:
            data_upload_output = json.load(f)

        training_file_id=data_upload_output['train_file_id']
        validation_file_id=data_upload_output['train_file_id']

        print("training file id : {}".format(training_file_id))
        print("validation file id : {}".format(validation_file_id))

        if training_file_id is not None:
            train_metadata = self._delete_file(training_file_id)
        else:
            logger.log("training file not present")
        
        if validation_file_id is not None:
            validation_metadata = self._delete_file(validation_file_id)
        else:
            logger.log("validation file not present")

        delete_files_output =  Dict[str, Dict[str, Any]] = {"train_file_id": train_metadata.id,
                                                          "validation_file_id": validation_metadata.id}
        
        return delete_files_output

    def _delete_file(self, file_id):
        delete_file_metadata = self.aoai_client.files.delete(file_id=file_id)
        delete_file_metadata = self.aoai_client.files.wait_for_processing(delete_file_metadata.id)
        self._check_delete_status(delete_file_metadata.model_dump())
        return delete_file_metadata

    def _check_delete_status(self, delete_file_metadata_dictionary: dict):
        """Check delete status of data."""
        file_id = delete_file_metadata_dictionary["id"]
        filename = delete_file_metadata_dictionary["filename"]
        delete_file_status = delete_file_metadata_dictionary["status"]

        if delete_file_status == "error":
            error_reason = delete_file_metadata_dictionary["error"]
            error_string = f"Deleting file failed for {filename}, file id: {file_id}, reason: {error_reason}"
            logger.error(error_string)
            raise Exception(error_string)

        logger.debug("file: {}, file id: {} deleted with status: {}".format(filename, file_id, delete_file_status))


def main():
    """Delete train and validation data to AOAI."""
    parser = argparse.ArgumentParser(description="Delete Component")
    parser.add_argument("--endpoint_name", type=str)
    parser.add_argument("--endpoint_resource_group", type=str)
    parser.add_argument("--endpoint_subscription", type=str)
    parser.add_argument("--data_upload_output", type=str)
    parser.add_argument("--data_delete_output", type=str)

    args = parser.parse_args()

    try:
        aoai_client_manager = AzureOpenAIClientManager(endpoint_name=args.endpoint_name,
                                                       endpoint_resource_group=args.endpoint_resource_group,
                                                       endpoint_subscription=args.endpoint_subscription)
        
        delete_component = DeleteComponent(aoai_client_manager.get_azure_openai_client())
        data_delete_output = delete_component.delete_files(args.data_upload_output)
        save_json(data_delete_output, args.data_delete_output)
        logger.info("deleted train and validation data")
        
    except Exception as e:
        logger.error("Got exception while running Delete data component. Ex: {}".format(e))
        raise e

if __name__ == "__main__":

    main()

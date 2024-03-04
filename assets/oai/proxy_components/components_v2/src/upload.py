# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# ---------------------------------------------------------

"""data upload component."""
from typing import Any, Dict
import argparse
from openai import AzureOpenAI
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.utils import save_json
from common.logging import get_logger

logger = get_logger(__name__)


class UploadComponent:
    """ Upload component to upload data to AOAI."""
    def __init__(self, aoai_client : AzureOpenAI):
        """ Upload component to upload data to AOAI."""
        self.aoai_client = aoai_client

    def upload_file(self, file_path):
        """ Upload file to AOAI."""
        file_data = open(file_path, "rb")
        upload_result = self.aoai_client.files.create(
            file=file_data,
            purpose='fine-tune')
        return upload_result

    def check_upload_status(self, upload_file_metadata_dictionary: dict):
        """ Check upload status."""
        file_id = upload_file_metadata_dictionary["id"]
        if upload_file_metadata_dictionary["status"] == "error":
            error_reason = upload_file_metadata_dictionary["error"]
            print("uploading file failed for file id: {}, reason: {}".format(file_id, error_reason))
            raise Exception("uploading file failed for file id: {}, reason: {}".format(file_id, error_reason))


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
        aoai_client = AzureOpenAIClientManager(endpoint_name=args.endpoint_name,
                                               endpoint_resource_group=args.endpoint_resource_group,
                                               endpoint_subscription=args.endpoint_subscription) \
            .get_azure_openai_client()
        upload_component = UploadComponent(aoai_client)

        # upload train data
        logger.debug("uploading train dataset")
        train_metadata = upload_component.upload_file(args.train_dataset)
        upload_component.check_upload_status(train_metadata.dict())
        logger.debug("uploaded train dataset, retrieved metadata : {}".format(train_metadata))

        dataset_upload_output : Dict[str, Dict[str, Any]] = {"train_file_id" : train_metadata.id}

        # upload validation data
        if args.validation_dataset is not None:
            logger.debug("uploading validation dataset")
            validation_metadata = upload_component.upload_file(args.validation_dataset)
            upload_component.check_upload_status(validation_metadata.dict())
            logger.debug("uploaded validation dataset, retrieved metadata : {}".format(validation_metadata))

            dataset_upload_output["validation_file_id"] = validation_metadata.id

        save_json(dataset_upload_output, args.dataset_upload_output)

    except Exception as e:
        logger.error("Got exception while running Upload data component. Ex: {}".format(e))
        print("Failed finetune - {}".format(e))
        raise e


if __name__ == "__main__":
    main()

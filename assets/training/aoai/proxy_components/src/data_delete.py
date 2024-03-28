# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""data delete component."""
import argparse
from openai import AzureOpenAI
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.logging import get_logger, add_custom_dimenions_to_app_insights_handler
import json

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

        training_file_id = data_upload_output['train_file_id']
        validation_file_id = data_upload_output['validation_file_id']

        if training_file_id is not None:
            self.aoai_client.files.delete(file_id=training_file_id)
            logger.debug("training file id: {} deleted".format(training_file_id))
        else:
            logger.log("training file not present")

        if validation_file_id is not None:
            self.aoai_client.files.delete(file_id=validation_file_id)
            logger.debug("validation file id: {} deleted".format(validation_file_id))
        else:
            logger.log("validation file not present")


def main():
    """Delete train and validation data to AOAI."""
    parser = argparse.ArgumentParser(description="Delete Component")
    parser.add_argument("--endpoint_name", type=str)
    parser.add_argument("--endpoint_resource_group", type=str)
    parser.add_argument("--endpoint_subscription", type=str)
    parser.add_argument("--data_upload_output", type=str)
    parser.add_argument("--wait_for_finetuning", type=str)
    args = parser.parse_args()

    try:
        aoai_client_manager = AzureOpenAIClientManager(endpoint_name=args.endpoint_name,
                                                       endpoint_resource_group=args.endpoint_resource_group,
                                                       endpoint_subscription=args.endpoint_subscription)
        add_custom_dimenions_to_app_insights_handler(logger,
                                                     aoai_client_manager.endpoint_name,
                                                     aoai_client_manager.endpoint_resource_group,
                                                     aoai_client_manager.endpoint_subscription)
        delete_component = DeleteComponent(aoai_client_manager.get_azure_openai_client())
        logger.info("deleting training and validataion data from Azure OpenAI")
        delete_component.delete_files(args.data_upload_output)

    except Exception as e:
        logger.error("Got exception while running delete data component. Ex: {}".format(e))
        raise e


if __name__ == "__main__":

    main()

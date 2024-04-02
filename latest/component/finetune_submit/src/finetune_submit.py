# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Finetune submit component."""

import argparse
import time
import json
from openai.types.fine_tuning.job_create_params import Hyperparameters
from common import utils
from common.cancel_handler import CancelHandler
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.logging import get_logger, add_custom_dimenions_to_app_insights_handler
from proxy_component import AzureOpenAIProxyComponent

logger = get_logger(__name__)


class FineTuneComponent(AzureOpenAIProxyComponent):
    """Fine-tune proxy class to submit fine-tune job to AOAI."""

    def __init__(self, aoai_client_manager: AzureOpenAIClientManager):
        """Fine-tune proxy class constructor."""
        super().__init__(aoai_client_manager.endpoint_name,
                         aoai_client_manager.endpoint_resource_group,
                         aoai_client_manager.endpoint_subscription)
        self.aoai_client = aoai_client_manager.get_azure_openai_client()

    def submit_finetune_job(self, training_file_id, validation_file_id, model, registered_model,
                            n_epochs, batch_size, learning_rate_multiplier, suffix=None):
        """Submit fine-tune job to AOAI."""
        logger.debug(f"Starting fine-tune job, model: {model}, n_epochs: {n_epochs},\
                     batch_size: {batch_size}, learning_rate_multiplier: {learning_rate_multiplier},\
                     training_file_id: {training_file_id}, validation_file_id: {validation_file_id}, suffix: {suffix}")
        hyperparameters: Hyperparameters = {
            "n_epochs": n_epochs if n_epochs else "auto",
            "batch_size": batch_size if batch_size else "auto",
            "learning_rate_multiplier": learning_rate_multiplier if learning_rate_multiplier else "auto"
        }
        finetune_job = self.aoai_client.fine_tuning.jobs.create(
            model=model,
            training_file=training_file_id,
            validation_file=validation_file_id,
            hyperparameters=hyperparameters,
            suffix=suffix)
        job_id = finetune_job.id
        status = finetune_job.status

        # If the job isn't yet done, poll it every 30 seconds.
        if status not in ["succeeded", "failed"]:
            logger.info(f'Job not in terminal status: {status}. Waiting.')
            while status not in ["succeeded", "failed"]:
                time.sleep(30)
                finetune_job = self.aoai_client.fine_tuning.jobs.retrieve(job_id)
                status = finetune_job.status
                logger.info(f"current status of finetune job : {status}, polling after 30 sec")
        else:
            logger.debug(f'Fine-tune job {job_id} finished with status: {status}')

        if status != "succeeded":
            raise Exception("Component failed")
        else:
            finetuned_model_id = finetune_job.fine_tuned_model
            logger.info(f'Fine-tune job {job_id} finished with \
                        status: {status}. model id: {finetuned_model_id}')
            return finetuned_model_id


def submit_finetune_job():
    """Submit fine-tune job to AOAI."""
    parser = argparse.ArgumentParser(description="Finetune submit Component")
    parser.add_argument("--model", type=str)
    parser.add_argument("--task_type", type=str)
    parser.add_argument("--registered_model_name", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate_multiplier", type=float)
    parser.add_argument("--data_upload_output", type=str)
    parser.add_argument("--finetune_submit_output", type=str)
    parser.add_argument("--endpoint_name", type=str)
    parser.add_argument("--endpoint_resource_group", type=str)
    parser.add_argument("--endpoint_subscription", type=str)

    args = parser.parse_args()
    logger.debug("args: {}".format(args))

    try:
        aoai_client_manager = AzureOpenAIClientManager(endpoint_name=args.endpoint_name,
                                                       endpoint_resource_group=args.endpoint_resource_group,
                                                       endpoint_subscription=args.endpoint_subscription)
        finetune_component = FineTuneComponent(aoai_client_manager)
        add_custom_dimenions_to_app_insights_handler(logger, finetune_component)
        CancelHandler.register_cancel_handler(finetune_component)

        logger.info("Starting finetune submit component")

        with open(args.data_upload_output) as f:
            data_upload_output = json.load(f)
        logger.debug(f"data_upload_output for finetuning model: {data_upload_output}")

        finetuned_model_id = finetune_component.submit_finetune_job(
            training_file_id=data_upload_output['train_file_id'],
            validation_file_id=data_upload_output['validation_file_id'],
            model=args.model,
            registered_model=args.registered_model_name,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate_multiplier=args.learning_rate_multiplier)

        utils.save_json({"finetuned_model_id": finetuned_model_id}, args.finetune_submit_output)
        logger.info("Completed finetune submit component")

    except Exception as e:
        logger.error("Got exception while running finetune submit component. Ex: {}".format(e))
        raise e


if __name__ == "__main__":
    submit_finetune_job()

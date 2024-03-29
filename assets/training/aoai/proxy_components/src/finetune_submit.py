# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Finetune submit component."""

import argparse
import time
import json
from openai import OpenAI
from openai.types.fine_tuning.job_create_params import Hyperparameters
from common import utils
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.logging import get_logger, add_custom_dimenions_to_app_insights_handler
import mlflow
import io
import pandas as pd


logger = get_logger(__name__)


class FineTuneProxy:
    """Fine-tune proxy class to submit fine-tune job to AOAI."""

    def __init__(self, aoai_client):
        """Fine-tune proxy class constructor."""
        self.aoai_client: OpenAI = aoai_client

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

        # If the job isn't yet done, poll it every 60 seconds.
        if status not in ["succeeded", "failed"]:
            logger.info(f'Job not in terminal status: {status}. Waiting.')
            last_event = None
            while status not in ["succeeded", "failed"]:
                time.sleep(60)
                finetune_job = self.aoai_client.fine_tuning.jobs.retrieve(job_id)
                status = finetune_job.status
                last_event = self._log_events(job_id, last_event)

        if status != "succeeded":
            error = finetune_job.error
            logger.error(f"Fine tuning job: {job_id} failed with error: {error}")
            raise Exception(f"Fine tuning job: {job_id} failed with error: {error}")

        finetuned_model_id = finetune_job.fine_tuned_model
        logger.info(f'Fine-tune job: {job_id} finished successfully. model id: {finetuned_model_id}')
        logger.info("fetching training metrics from Azure OpenAI")
        self._log_metrics(job_id)
        return finetuned_model_id
        
    def _log_metrics(self, job_id):
        result_files = self.aoai_client.fine_tuning.jobs.retrieve(job_id).result_files
        response = self.aoai_client.files.content(file_id=result_files)
        f = io.BytesIO(response.content)
        df = pd.read_csv(f)

        for col in df.columns[1:]:
            values = df[['step', col]].dropna()
            # drop all rows with -1 as a value
            values = values[values[col] != -1]
            for i, row in values.iterrows():
                mlflow.log_metric(key=col, value=row[col], step=int(row.step))

    def _log_events(self, job_id, last_event):
        job_events = self.aoai_client.fine_tuning.jobs.list_events(job_id).data
        event_message_list = []
        for job_event in job_events:
            if last_event == job_event.message:
                break
            event_message_list.append(job_event.message)
        
        if len(event_message_list) > 0:
            last_event = event_message_list[0]
            event_message_list.reverse()
            for event_message in event_message_list:
                logger.debug(event_message)
        return last_event


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
        add_custom_dimenions_to_app_insights_handler(logger,
                                                     aoai_client_manager.endpoint_name,
                                                     aoai_client_manager.endpoint_resource_group,
                                                     aoai_client_manager.endpoint_subscription)
        with open(args.data_upload_output) as f:
            data_upload_output = json.load(f)
        logger.debug(f"data_upload_output for finetuning model: {data_upload_output}")
        finetune_proxy = FineTuneProxy(aoai_client_manager.get_azure_openai_client())
        fientuned_model_id = finetune_proxy.submit_finetune_job(
            training_file_id=data_upload_output['train_file_id'],
            validation_file_id=data_upload_output['validation_file_id'],
            model=args.model,
            registered_model=args.registered_model_name,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate_multiplier=args.learning_rate_multiplier)

        utils.save_json({"finetuned_model_id": fientuned_model_id}, args.finetune_submit_output)

    except Exception as e:
        logger.error("Got exception while running Finetune submit component. Ex: {}".format(e))
        print("Failed finetune - {}".format(e))
        raise e


if __name__ == "__main__":
    submit_finetune_job()

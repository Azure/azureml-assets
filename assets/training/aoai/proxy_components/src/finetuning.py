# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Azure Open AI Finetuning component."""

import argparse
import time
from typing import Optional
from openai.types.fine_tuning.job_create_params import Hyperparameters
from common import utils
from common.cancel_handler import CancelHandler
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.logging import get_logger, add_custom_dimenions_to_app_insights_handler
from proxy_component import AzureOpenAIProxyComponent
import mlflow
import io
import pandas as pd


logger = get_logger(__name__)

class AzureOpenAIFinetuning(AzureOpenAIProxyComponent):
    """Fine-tune proxy class to submit fine-tune job to AOAI."""

    def __init__(self, aoai_client_manager: AzureOpenAIClientManager):
        super().__init__(aoai_client_manager.endpoint_name,
                    aoai_client_manager.endpoint_resource_group,
                    aoai_client_manager.endpoint_subscription)
        self.aoai_client = aoai_client_manager.get_azure_openai_client()
        self.training_file_id = None
        self.validation_file_id = None
        self.finetuning_job_id = None

    def submit_job(self, training_file_path: str, validation_file_path: Optional[str], model: str, n_epochs: Optional[int],
                    batch_size: Optional[int], learning_rate_multiplier: Optional[float], suffix=Optional[str]):
        
        logger.info("Step 1: Uploading data to AzureOpenAI resource")
        self.upload_files(training_file_path, validation_file_path)

        logger.info("Step 2: Finetuning model")
        finetuned_model_id = self.submit_finetune_job(model, n_epochs, batch_size, learning_rate_multiplier, suffix)
        finetuned_job = self.track_finetuning_job()

        logger.info("Step 3: Deleting data from AzureOpenAI resource")
        self.delete_files()

        if finetuned_job.status == "failed":
            raise Exception(f"Fine tuning job: {self.finetuning_job_id} failed with error: {finetuned_job.error}")
        elif finetuned_job == "cancelled":
            logger.info(f"finetune job: {self.finetuning_job_id} got cancelled")
            return None
        else:
            logger.info(f'Fine-tune job: {self.finetuning_job_id} finished successfully')

        return finetuned_model_id

    def delete_files(self):
        if self.training_file_id is not None:
            self.aoai_client.files.delete(file_id=self.training_file_id)
            self._wait_for_processing(self.training_file_id)
            logger.debug("training file id: {} deleted".format(self.training_file_id))
        
        if self.validation_file_id is not None:
            self.aoai_client.files.delete(file_id=self.validation_file_id)
            self._wait_for_processing(self.validation_file_id)
            logger.debug("validation file id: {} deleted".format(self.validation_file_id))
        
    def upload_files(self, train_file_path: str, validation_file_path: str = None):
        """Upload training and validation files to azure openai."""
        
        train_file_name, validation_file_name = utils.get_train_validation_filename(train_file_path, validation_file_path)
        train_data, validation_data = utils.get_train_validation_data(train_file_path, validation_file_path)

        if validation_file_path is None:
            logger.debug(f"validation file not provided, train data will be split in {utils.train_dataset_split_ratio} ratio to create validation data")

        logger.debug(f"uploading training file : {train_file_name}")
        train_metadata = self.aoai_client.files.create(file=(train_file_name, train_data, 'application/json'), purpose='fine-tune')
        self.training_file_id = train_metadata.id
        self._wait_for_processing(train_metadata.id)
        logger.info("training file uploaded")

        logger.debug(f"uploading validation file : {validation_file_name}")
        validation_metadata = self.aoai_client.files.create(file=(validation_file_name, validation_data, 'application/json'), purpose='fine-tune')
        self.validation_file_id = validation_metadata.id
        self._wait_for_processing(validation_metadata.id)
        logger.info("validation file uploaded")

    def _wait_for_processing(self, file_id):
        upload_file_metadata = self.aoai_client.files.wait_for_processing(file_id)
        logger.info(f"file status is : {upload_file_metadata.status} for file name : {upload_file_metadata.filename}")

        if upload_file_metadata.status == "error":
            error_reason = upload_file_metadata.model_dump()["error"]
            error_string = f"Processing file failed for {upload_file_metadata.filename}, file id: {upload_file_metadata.id}, reason: {error_reason}"
            logger.error(error_string)
            raise Exception(error_string)

    def submit_finetune_job(self,model, n_epochs, batch_size, learning_rate_multiplier, suffix=None):
        """Submit fine-tune job to AOAI."""
        logger.debug(f"Starting fine-tune job, model: {model}, n_epochs: {n_epochs},\
                     batch_size: {batch_size}, learning_rate_multiplier: {learning_rate_multiplier},\
                     training_file_id: {self.training_file_id}, validation_file_id: {self.validation_file_id}, suffix: {suffix}")
        hyperparameters = self.get_hyperparameters_dict(n_epochs, batch_size, learning_rate_multiplier)

        finetune_job = self.aoai_client.fine_tuning.jobs.create(
            model=model,
            training_file=self.training_file_id,
            validation_file=self.validation_file_id,
            hyperparameters=hyperparameters,
            suffix=suffix)
        
        logger.debug(f"started finetuning job in Azure OpenAI resource. Job id: {finetune_job.id}")

        self.finetuning_job_id = finetune_job.id
        return finetune_job.fine_tuned_model

    def track_finetuning_job(self):
        finetune_job = self.aoai_client.fine_tuning.jobs.retrieve(self.finetuning_job_id)
        job_status = finetune_job.status

        terminal_statuses = ["succeeded", "failed", "cancelled"]
        last_metric_logged = 0
        last_event_message = None

        while job_status not in terminal_statuses:
            time.sleep(60)
            finetune_job = self.aoai_client.fine_tuning.jobs.retrieve(self.finetuning_job_id)
            job_status = finetune_job.status
            logger.info(f"Finetuning job status : {job_status}")
            last_event_message = self._log_events(last_event_message)

            if finetune_job.result_files is not None:
                last_metric_logged = self._log_metrics(finetune_job, last_metric_logged)

        # emitting metrics after status becomes terminal, just to ensure we do not miss any metric
        finetune_job = self.aoai_client.fine_tuning.jobs.retrieve(self.finetuning_job_id)
        last_metric_logged = self._log_metrics(finetune_job, last_metric_logged)

        return finetune_job

    def cancel_job(self):
        """Cancel finetuning job in Azure OpenAI resource."""
        logger.debug(f"job cancellation has been triggered, cancelling finetuning job")

        self.delete_files()

        if self.finetuning_job_id is not None:
            logger.info("cancelling finetuning job in AzureOpenAI resource")
            self.aoai_client.fine_tuning.jobs.cancel(self.finetuning_job_id)
        else:
            logger.debug("finetuning job not created, not starting now as cancellation is triggered")

        exit()

    def _log_metrics(self, finetune_job, last_metric_logged):
        """Fetch training metrics from azure open ai resource after finetuning is done and log them."""
        result_file = finetune_job.result_files[0]
        if result_file is None:
            logger.warning("result file for the finetuning job not present, cannot log training metrics")
            return

        response = self.aoai_client.files.content(file_id=result_file)
        if response is None or response.content is None:
            logger.warning("content not present in result file for the job, cannot log training metrics")
            return
        f = io.BytesIO(response.content)
        df = pd.read_csv(f)

        if last_metric_logged >= len(df):
            logger.info(f"no new metrics emitted since last iteration,\
                        currently metrics logged till {last_metric_logged} steps")
            return last_metric_logged

        for col in df.columns[1:]:
            values = df[['step', col]].dropna()
            # drop all rows with -1 as a value
            values = values[values[col] != -1]
            values = values.iloc[last_metric_logged:]
            for i, row in values.iterrows():
                mlflow.log_metric(key=col, value=row[col], step=int(row.step))
        last_metric_logged = len(df)
        logger.info(f"logged training metrics for finetuning job till {last_metric_logged} steps")
        return last_metric_logged

    def _log_events(self, last_event_message):
        """Log events like training started, running nth epoch etc."""
        job_events = self.aoai_client.fine_tuning.jobs.list_events(self.finetuning_job_id).data
        events_message_list = utils.list_event_messages_after_given_event(job_events, last_event_message)

        if len(events_message_list) > 0:
            for message in events_message_list:
                logger.debug(message)
            return events_message_list[-1]
        
        return last_event_message

    def get_hyperparameters_dict(self, n_epochs, batch_size, learning_rate_multiplier) -> Hyperparameters:
        
        hyperparameters: Hyperparameters = {}

        if n_epochs:
            hyperparameters["n_epochs"] = n_epochs
        else:
            logger.info("num epochs not passed, it will be determined dynamically")

        if batch_size:
            hyperparameters["batch_size"] = batch_size
        else:
            logger.info("batch size not passed, it will be determined dynamically")

        if learning_rate_multiplier:
            hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier
        else:
            logger.info("learning rate multiplier not passed, it will be determined dynamically")
        
        return hyperparameters
        """Log events like training started, running nth epoch etc."""
        job_events = self.aoai_client.fine_tuning.jobs.list_events(self.job_id).data
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


def main():
    """Submit fine-tune job to AOAI."""
    parser = argparse.ArgumentParser(description="AOAI Finetuning Component")
    parser.add_argument("--training_file_path", type=str)
    parser.add_argument("--validation_file_path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--task_type", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate_multiplier", type=float)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--aoai_finetuning_output", type=str)
    parser.add_argument("--endpoint_name", type=str)
    parser.add_argument("--endpoint_resource_group", type=str)
    parser.add_argument("--endpoint_subscription", type=str)

    args = parser.parse_args()
    logger.debug("args: {}".format(args))

    try:
        aoai_client_manager = AzureOpenAIClientManager(endpoint_name=args.endpoint_name,
                                                       endpoint_resource_group=args.endpoint_resource_group,
                                                       endpoint_subscription=args.endpoint_subscription)
        finetune_component = AzureOpenAIFinetuning(aoai_client_manager)
        add_custom_dimenions_to_app_insights_handler(logger, finetune_component)
        CancelHandler.register_cancel_handler(finetune_component)

        logger.info("Starting Finetuning in Azure OpenAI resource")

        finetuned_model_id = finetune_component.submit_job(
            training_file_path = args.training_file_path,
            validation_file_path = args.validation_file_path,
            model=args.model,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate_multiplier=args.learning_rate_multiplier,
            suffix=args.suffix
        )

        utils.save_json({"finetuned_model_id": finetuned_model_id}, args.finetune_submit_output)
        logger.info("Completed finetuning in Azure OpenAI resource")

    except SystemExit:
        logger.warning("Exiting finetuning job")
    except Exception as e:
        logger.error("Got exception while finetuning. Ex: {}".format(e))
        raise e


if __name__ == "__main__":
    main()
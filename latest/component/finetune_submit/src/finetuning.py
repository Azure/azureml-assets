# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Azure Open AI Finetuning component."""

import argparse
from argparse import Namespace
import time
from typing import Optional, Dict
from common import utils
from common.cancel_handler import CancelHandler
from common.azure_openai_client_manager import AzureOpenAIClientManager
from common.keyvault_client_manager import KeyVaultClientManager
from common.logging import get_logger, add_custom_dimenions_to_app_insights_handler
from proxy_component import AzureOpenAIProxyComponent
from hyperparameters import Hyperparameters, Hyperparameters_1P
import mlflow
import io
import pandas as pd


logger = get_logger(__name__)
terminal_statuses = ["succeeded", "failed", "cancelled"]


class AzureOpenAIFinetuning(AzureOpenAIProxyComponent):
    """Fine-tune class to submit fine-tune job to AOAI."""

    def __init__(self, aoai_client_manager: AzureOpenAIClientManager):
        """Fine-tune class to submit fine-tune job to AOAI."""
        super().__init__(aoai_client_manager.endpoint_name,
                         aoai_client_manager.endpoint_resource_group,
                         aoai_client_manager.endpoint_subscription)
        self.aoai_client = aoai_client_manager.aoai_client
        self.aoai_client_manager = aoai_client_manager
        self.training_file_id = None
        self.validation_file_id = None
        self.finetuning_job_id = None

    def submit_job(self, training_file_path: Optional[str], validation_file_path: Optional[str],
                   training_import_path: Optional[str], validation_import_path: Optional[str], model: str,
                   hyperparameters: Dict[str, str], hyperparameters_1p: Dict[str, str], suffix=Optional[str]) -> str:
        """Upload data, finetune model and then delete data."""
        try:
            logger.info("Step 1:Uploading data to AzureOpenAI resource")
            if training_file_path is not None:
                self.upload_files(training_file_path, validation_file_path)
            else:
                training_data_uri_key, training_data_uri = utils.get_key_or_uri_from_data_import_path(
                    training_import_path
                )
                if validation_import_path is not None:
                    validation_data_uri_key, validation_data_uri =\
                        utils.get_key_or_uri_from_data_import_path(validation_import_path)

                if training_data_uri_key is not None:
                    keyvault_client_manager = KeyVaultClientManager()
                    keyvault_client = keyvault_client_manager.get_keyvault_client()
                    logger.info(f"fetching training file uri from keyvault : {keyvault_client_manager.keyvault_name}")
                    training_data_uri = keyvault_client.get_secret(training_data_uri_key).value
                else:
                    logger.info("User has provided trainining data uri directly, sending it to Azure OpenAI resource")

                self.training_file_id = self.upload_file_uri_from_rest(training_data_uri)
                logger.info("uploaded training file uri to aoai resource")

                if validation_import_path is not None:
                    if validation_data_uri_key is not None:
                        keyvault_client_manager = KeyVaultClientManager()
                        keyvault_client = keyvault_client_manager.get_keyvault_client()
                        logger.info(
                            f"fetching validation fileuri from keyvault: {keyvault_client_manager.keyvault_name}"
                        )
                        validation_data_uri = keyvault_client.get_secret(validation_data_uri_key).value
                    else:
                        logger.info("User provided validation data uri directly, sending it to Azure OpenAI resource")

                    self.validation_file_id = self.upload_file_uri_from_rest(validation_data_uri)
                    logger.info("uploaded validation file uri to aoai resource")

            logger.info("Step 2: Finetuning model")
            self.finetuning_job_id = self.submit_finetune_job(model, hyperparameters, hyperparameters_1p, suffix)
            finetuned_job = self.track_finetuning_job()

            logger.debug(f"Finetuned model name: {finetuned_job.fine_tuned_model}, status: {finetuned_job.status}")
            logger.info("Step 3: Deleting data from AzureOpenAI resource")
            self.delete_files()

            if finetuned_job.status == "failed":
                raise Exception(f"Fine tuning job: {self.finetuning_job_id} failed with error: {finetuned_job.error}")
            elif finetuned_job.status == "cancelled":
                logger.info(f"finetune job: {self.finetuning_job_id} got cancelled")
                return None
            else:
                logger.info(f'Fine-tune job: {self.finetuning_job_id} finished successfully')

            return finetuned_job.fine_tuned_model, self.finetuning_job_id
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.info("Deleting data from AzureOpenAI resource")
            self.delete_files()
            raise e

    def delete_files(self):
        """Delete training and validation files from azure openai resource."""
        if self.training_file_id is not None:
            self.aoai_client.files.delete(file_id=self.training_file_id)
            logger.debug(f"file id: {self.training_file_id} deleted")

        if self.validation_file_id is not None:
            self.aoai_client.files.delete(file_id=self.validation_file_id)
            logger.debug(f"file id: {self.validation_file_id} deleted")

    def upload_files(self, train_file_path: str, validation_file_path: str = None):
        """Upload training and validation files to azure openai resource."""
        train_file_name, validation_file_name = utils.get_train_validation_filename(train_file_path,
                                                                                    validation_file_path)
        train_data, validation_data = utils.get_train_validation_data(train_file_path, validation_file_path)

        if validation_file_path is None:
            logger.debug(f"validation file not provided, train data will be split in\
                         {utils.train_dataset_split_ratio} ratio to create validation data")

        logger.debug(f"uploading training file : {train_file_name}")
        train_metadata = self.aoai_client.files.create(file=(train_file_name, train_data, 'application/json'),
                                                       purpose='fine-tune')
        self.training_file_id = train_metadata.id
        self._wait_for_processing(train_metadata.id)
        logger.info("training file uploaded")

        logger.debug(f"uploading validation file : {validation_file_name}")
        validation_metadata = self.aoai_client.files.create(file=(validation_file_name,
                                                            validation_data, 'application/json'),
                                                            purpose='fine-tune')
        self.validation_file_id = validation_metadata.id
        self._wait_for_processing(validation_metadata.id)
        logger.info("validation file uploaded")

    def upload_file_uri_from_rest(self, file_uri: str) -> str:
        """Upload file uri to azure openai resource via rest call."""
        file_uri_payload = utils.create_payload_for_data_upload_rest_call(file_uri)
        file_upload_response = self.aoai_client_manager.upload_data_to_aoai(file_uri_payload)

        file_id = utils.parse_file_id_from_upload_response(file_upload_response)
        self._wait_for_processing(file_id)
        logger.info(f"file id : {file_id} uploaded")

        return file_id

    def _wait_for_processing(self, file_id):
        upload_file_metadata = self.aoai_client.files.wait_for_processing(file_id)
        filename = upload_file_metadata.filename
        logger.info(f"file status is : {upload_file_metadata.status} for file name : {filename}")

        if upload_file_metadata.status == "error":
            error_reason = upload_file_metadata.status_details
            error_string = f"Processing file failed for {filename},\
                           file id: {upload_file_metadata.id}, reason: {error_reason}"
            logger.error(error_string)
            raise Exception(error_string)

    def submit_finetune_job(self, model, hyperparameters: Dict[str, str],
                            hyperparameters_1p: Dict[str, str], suffix=None):
        """Submit fine-tune job to AOAI."""
        logger.debug(f"Starting fine-tune job, model: {model}, suffix: {suffix},\
                     training_file_id: {self.training_file_id}, validation_file_id: {self.validation_file_id}")

        finetune_job = self.aoai_client.fine_tuning.jobs.create(
            model=model,
            training_file=self.training_file_id,
            validation_file=self.validation_file_id,
            hyperparameters=hyperparameters,
            extra_headers=hyperparameters_1p,
            suffix=suffix)

        logger.debug(f"started finetuning job in Azure OpenAI resource. Job id: {finetune_job.id}")

        return finetune_job.id

    def track_finetuning_job(self):
        """Fetch metrics for the job and log them."""
        finetune_job = self.aoai_client.fine_tuning.jobs.retrieve(self.finetuning_job_id)
        job_status = finetune_job.status

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
        logger.debug("job cancellation has been triggered, cancelling job")

        if self.finetuning_job_id is not None:
            logger.info("cancelling finetuning job in AzureOpenAI resource")
            cancelled_job = self.aoai_client.fine_tuning.jobs.cancel(self.finetuning_job_id)
            while cancelled_job.status not in terminal_statuses:
                time.sleep(10)
                logger.debug(f"cancellation triggered, job not cancelled yet, current status: {cancelled_job.status}")
                cancelled_job = self.aoai_client.fine_tuning.jobs.retrieve(cancelled_job.id)
        else:
            logger.debug("finetuning job not created, not starting now as cancellation is triggered")

        self.delete_files()
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


def parse_args():
    """Parse arguements provided to the component."""
    parser = argparse.ArgumentParser(description="AOAI Finetuning Component")
    parser.add_argument("--endpoint_name", type=str)
    parser.add_argument("--endpoint_resource_group", type=str)
    parser.add_argument("--endpoint_subscription", type=str)

    parser.add_argument("--training_file_path", type=str)
    parser.add_argument("--validation_file_path", type=str)

    parser.add_argument("--training_import_path", type=str)
    parser.add_argument("--validation_import_path", type=str)
    parser.add_argument("--aoai_finetuning_output", type=str)

    parser.add_argument("--model", type=str)
    parser.add_argument("--task_type", type=str)
    parser.add_argument("--suffix", type=str)

    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate_multiplier", type=float)

    parser.add_argument("--lora_dim", type=int)
    parser.add_argument("--n_ctx", type=int)
    parser.add_argument("--weight_decay_multiplier", type=float)

    args = parser.parse_args()
    return args


def validate_train_data_upload_type(args: Namespace) -> str:
    """Validate input of dataset."""
    if args.training_file_path is None and args.training_import_path is None:
        raise ValueError("One of training file path or training import path should be provided")

    if args.training_file_path is not None and args.training_import_path is not None:
        raise ValueError("Exactly one of training file path or training import path must be provided")

    if args.validation_file_path is not None and args.validation_import_path is not None:
        raise ValueError("Exactly one of validation file path and validation import path must be provided")


def main():
    """Submit fine-tune job to AOAI."""
    args = parse_args()
    logger.debug("job args: {}".format(args))

    try:
        aoai_client_manager = AzureOpenAIClientManager(endpoint_name=args.endpoint_name,
                                                       endpoint_resource_group=args.endpoint_resource_group,
                                                       endpoint_subscription=args.endpoint_subscription)
        finetune_component = AzureOpenAIFinetuning(aoai_client_manager)
        add_custom_dimenions_to_app_insights_handler(logger, finetune_component)
        CancelHandler.register_cancel_handler(finetune_component)

        logger.info("Starting Finetuning in Azure OpenAI resource")

        hyperparameters = Hyperparameters(**vars(args))
        logger.debug("hyperparameters: {}".format(hyperparameters))

        hyperparameters_1p = Hyperparameters_1P(**vars(args))
        logger.debug("hyperparameters for 1P: {}".format(hyperparameters_1p))

        validate_train_data_upload_type(args)

        finetuned_model_id, finetune_job_id = finetune_component.submit_job(
            training_file_path=args.training_file_path,
            validation_file_path=args.validation_file_path,
            training_import_path=args.training_import_path,
            validation_import_path=args.validation_import_path,
            model=args.model,
            hyperparameters=hyperparameters.get_dict(),
            hyperparameters_1p=hyperparameters_1p.get_dict(),
            suffix=args.suffix
        )

        utils.save_json(
            {"finetuned_model_id": finetuned_model_id, "finetune_job_id": finetune_job_id},
            args.aoai_finetuning_output
        )
        logger.info("Completed finetuning in Azure OpenAI resource")

    except SystemExit:
        logger.warning("Exiting finetuning job")
    except Exception as e:
        logger.error("Got exception while finetuning. Ex: {}".format(e))
        raise e


if __name__ == "__main__":
    main()

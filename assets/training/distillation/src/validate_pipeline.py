# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for validating distillation pipeline arguments."""
import logging
import requests
import pandas as pd
from argparse import ArgumentParser, Namespace

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.telemetry.activity import log_activity
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)

from generate_data import get_parser

from common.constants import (
    DataGenerationTaskType
)

from common.utils import (
    get_workspace_mlclient,
    get_base_url
)

from common.validation import (
    validate_file_paths_with_supported_formats,
    validate_file_exists,
    validate_model_temperature,
    validate_model_top_p,
    validate_model_frequency_penalty,
    validate_model_presence_penalty,
)

logger = get_logger_app("azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import")

COMPONENT_NAME = "oss_distillation_validate_pipeline"

def update_finetuning_parser(parser: ArgumentParser):
    """
    Updates parser with flags from finetuning task as part of the distillation 
    pipeline.
    """
    # TODO (nandakumars): add relevant finetuning arguments.
    return parser

class PipelineInputsValidator:
    def __init__(self, args: Namespace) -> None:
        self._args = args
        with log_activity(logger=logger, activity_name="ML_CLIENT_INITIALISATION"):
            ws_mlclient = get_workspace_mlclient()
            if not ws_mlclient:
                raise Exception("Could not create MLClient for current workspace")
            self._mlclient = ws_mlclient
        
        with log_activity(logger=logger, activity_name="VALIDATE_DATA_GENERATION_INPUTS"):
            self._validate_data_generation_inputs()
    
    def _validate_model_endpoint_args(self):

        if self._args.teacher_model_endpoint_name:
            # This block should populate endpoint url & key, if not present already.
            pass

        if not self._args.teacher_model_endpoint_url \
        or not self._args.teacher_model_endpoint_key:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "Endpoint URL and key are required fields for data generation."
                    )
                )
            )
    
    def _validate_model_endpoint(self):
        """Validates model endpoints availability by retrieving its details."""
        base_url = get_base_url(self._args.teacher_model_endpoint_url)
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._args.teacher_model_endpoint_key}"
        }

        # https://learn.microsoft.com/en-us/azure/machine-learning/reference-model-inference-info
        response = requests.get(url=f"{base_url}/info", headers=request_headers)
        response.raise_for_status()
        response_data = response.json()
        model_name = response_data.get("model_name")
        logger.info(f"Model validated, model name - {model_name}")

    def _validate_inference_parameters(self):
        """Validates all body parameters passed as part of inference."""
        validate_model_temperature(self._args.teacher_model_temperature)
        validate_model_top_p(self._args.teacher_model_top_p)
        validate_model_presence_penalty(self._args.teacher_model_presence_penalty)
        validate_model_frequency_penalty(self._args.teacher_model_frequency_penalty)

        # TODO (nandakumars): validate batch size & success ratio.

    def _validate_record_by_task(self, record: list) -> dict:
        """
        Validates record in a dataset against the data generation task type.
        Returns a dictionary containing exception if any validation error is found.

        Args:
            record (list): Sequence of messages
        """
        task_type = self._args.data_generation_task_type
        if task_type == DataGenerationTaskType.CONVERSATION and \
        len(record) < 3:
            return { "exception": f"Dataset needs to be of multi-turn for task type {DataGenerationTaskType.CONVERSATION}." }

        if task_type != DataGenerationTaskType.CONVERSATION and \
        len(record) > 2:
            return { "exception": f"Chat cannot be of type multi-turn." }

    def _validate_message(self, id: int, message: dict) -> dict:
        """
        Validates individual message in the dataset. Returns dictionary containing exception,
        if any validation error is found.

        Args:
            id (int): id of the message in sequence of messages.
            message (dict): Message object in sequence of messages. 
        """
        allowed_roles = ["system", "user", "assistant"]
        if 'role' not in message:
            return f"Message at index {id} is missing 'role'."
        
        if message['role'] not in allowed_roles:
            return f"Invalid 'role' at index {id}."

        if 'content' not in message:
            return f"Message at index {id} is missing 'content'."
        
    def _validate_record_content(self, record: list) -> dict:
            """
            Validates content of a record and ensures messages are in the expected format.
            Returns dictionary containing exception, if any validation error is found.

            Args: 
                record (list): Sequence of messages
            """
            try:
                if record[0].get("role") != "system":
                    role = record[0].get("role")
                    return {
                        "exception": f"First message should be of role 'system' but got {role}."
                    }
                    
                expected_roles = ["user", "assistant"]
                for id, message in enumerate(record[1:], start=1):
                    if not isinstance(message, dict):
                        return {
                            "exception": f"Message at index {id} should be a dictionary."
                        }
                    
                    err = self._validate_message(id=id, message=message)
                    if err:
                        return {
                            "exception": err
                        }
                    
                    expected_role = expected_roles[(id - 1) % 2]
                    if message.get("role") != expected_role:
                        return {
                            "exception": f"Role at index {id} should be {expected_role}."
                        }
            except Exception as e:
                return {
                    "exception": e
                }
            
    def _validate_dataset_record(self, record: list) -> str:
        """Validates a record in the dataset. Returns the validation error if found.
        
        Args:
            record (list): Sequence of messages
        """
        if not record:
            return f"Chat cannot be empty."
        
        err = self._validate_record_by_task(record=record)
        if err and ("exception" in err):
            return err["exception"]

        err = self._validate_record_content(record=record)
        if err and ("exception" in err):
            return err["exception"]

    def _validate_dataset(self, file_path: str):
        """Validates training/validation dataset passed to the data-generation component.
        
        Args:
            file_path (str): Path to the dataset
        
        Raises:
            ACFTUserError: If a known validation error is caught
        """
        df = pd.read_json(file_path, lines=True, chunksize=self._args.request_batch_size)
        for batch in df:
            for idx, row in batch.iterrows():
                record = row.iloc[0]
                err = self._validate_dataset_record(record=record)
                if err:
                    raise ACFTValidationException._with_error(
                        AzureMLError.create(
                            ACFTUserError,
                            pii_safe_message=(
                                f"Error validating dataset record {idx}: {err}"
                            )
                        )
                    )

    def _validate_data_generation_inputs(self):
        """Validate all input flags to the data-generation component."""  
        with log_activity(logger=logger, activity_name="VALIDATE_FILE_PATH"):
            files = [self._args.train_file_path, self._args.validation_file_path]
            validate_file_paths_with_supported_formats(file_paths=files)
            validate_file_exists(file_paths=files)

        with log_activity(logger=logger, activity_name="VALIDATE_TEACHER_MODEL_ENDPOINT"):
            self._validate_model_endpoint_args()
            self._validate_model_endpoint()
        
        with log_activity(logger=logger, activity_name="VALIDATE_INFERENCE_PARAMETERS"):
            self._validate_inference_parameters()
        
        with log_activity(logger=logger, activity_name="VALIDATE_TRAINING_DATA"):
            self._validate_dataset(self._args.train_file_path)
        
        if self._args.validation_file_path:
            with log_activity(logger=logger, activity_name="VALIDATE_VALIDATION_DATA"):
                self._validate_dataset(self._args.validation_file_path)

@swallow_all_exceptions(time_delay=5)
def main():
    
    # Get data generation component input parameters.
    parser = get_parser()
    parser = update_finetuning_parser(parser=parser)
    args, _ = parser.parse_known_args()
    
    set_logging_parameters(
        task_type="DistillationPipelineValidation",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO
    )

    with log_activity(logger=logger, activity_name="VALIDATOR"):
        PipelineInputsValidator(args=args)

if __name__ == "__main__":
    main()
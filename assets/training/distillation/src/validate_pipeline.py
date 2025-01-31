# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for validating distillation pipeline arguments."""
import logging
import requests
import pandas as pd
import json
from argparse import Namespace
from pathlib import Path
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import (
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
)
from azureml.acft.common_components import (
    get_logger_app,
    set_logging_parameters,
    LoggingLiterals,
)
from azureml.acft.common_components.utils.error_handling.exceptions import (
    ACFTValidationException,
)
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    ACFTUserError,
)
from azureml.telemetry.activity import log_activity
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)

from generate_data import get_parser

from common.constants import (
    DataGenerationTaskType,
    TelemetryConstants,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    VLLM_CHAT_SCORE_PATH,
    MIN_RECORDS_FOR_FT,
    MATH_MIN_RECORDS_FOR_FT
)

from common.utils import (
    get_endpoint_details,
    get_workspace_mlclient,
    get_base_url,
    validate_teacher_model_details,
    exponential_backoff,
)

from common.validation import (
    validate_file_paths_with_supported_formats,
    validate_file_exists,
    validate_model_temperature,
    validate_model_top_p,
    validate_model_frequency_penalty,
    validate_model_presence_penalty,
    validate_request_batch_size,
    validate_min_endpoint_success_ratio,
)

logger = get_logger_app(
    "azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import"
)

COMPONENT_NAME = "oss_distillation_validate_pipeline"


class PipelineInputsValidator:
    """Dataclass for validating inputs to distillation pipeline."""

    def __init__(self, args: Namespace) -> None:
        """Initialise validator.

        Args:
            args (Namespace): Inputs flags to validate.
        """
        self._args = args
        with log_activity(
            logger=logger, activity_name=TelemetryConstants.ML_CLIENT_INITIALISATION
        ):
            ws_mlclient = get_workspace_mlclient()
            if not ws_mlclient:
                raise Exception("Could not create MLClient for current workspace")
            self._mlclient = ws_mlclient

        with log_activity(
            logger=logger,
            activity_name=TelemetryConstants.VALIDATE_DATA_GENERATION_INPUTS,
        ):
            self._validate_data_generation_inputs()

    def _get_dataframe(self, file_path: str):
        if not Path(file_path).is_file():
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"File not found at {file_path}. Please provide a valid file path."
                    ),
                )
            )
        try:
            return pd.read_json(
                file_path, lines=True, chunksize=self._args.request_batch_size
            )
        except ValueError as e:
            # If the file is not present pandas will read it as jsonl string
            # raises a ValueError if it is not a valid jsonl string.
            # also raises value error if it is not a valid jsonl file.
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"Error while reading JSON file. Make sure the file is a valid jsonl file. Error: {e}"
                    ),
                )
            )
        except Exception as e:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"An unexpected error occurred while reading the file: {e}"
                    ),
                )
            )

    def _get_inference_request_headers(self) -> dict:
        key = self._args.teacher_model_endpoint_key
        return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    def _get_cot_status(self) -> bool:
        cot_enabled = self._args.enable_chain_of_thought
        return cot_enabled.lower() == "true"

    def _get_cod_status(self) -> bool:
        cod_enabled = self._args.enable_chain_of_density
        return cod_enabled.lower() == "true"

    def _get_max_len_summary(self) -> bool:
        return self._args.max_len_summary != 80

    def _validate_model_endpoint_args(self):
        endpoint_name = self._args.teacher_model_endpoint_name
        if endpoint_name:
            endpoint_details = get_endpoint_details(
                mlclient_ws=self._mlclient, endpoint_name=endpoint_name
            )
            self._args.teacher_model_endpoint_url = endpoint_details.get_endpoint_url()
            self._args.teacher_model_endpoint_key = endpoint_details.get_endpoint_key()
            model_asset_id = endpoint_details.get_deployed_model_id()
            validate_teacher_model_details(model_asset_id)

        if (
            not self._args.teacher_model_endpoint_url
            or not self._args.teacher_model_endpoint_key
        ):
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "Endpoint URL and key are required fields for data generation."
                    ),
                )
            )

    @exponential_backoff()
    def _validate_model_endpoint(self):
        """Validate model endpoints availability by retrieving its details."""
        base_url = get_base_url(self._args.teacher_model_endpoint_url)
        request_headers = self._get_inference_request_headers()

        # https://learn.microsoft.com/en-us/azure/machine-learning/reference-model-inference-info
        response = requests.get(url=f"{base_url}/info", headers=request_headers)
        response.raise_for_status()
        response_data = response.json()
        model_name = response_data.get("model_name")
        logger.info(f"Model validated, model name - {model_name}")

    @exponential_backoff()
    def _validate_model_inference(self):
        """Validate a sample inference call.

        Raises:
            HTTPError: If one occured.
        """
        # Prep data.
        df = self._get_dataframe(file_path=self._args.train_file_path)
        batch = next(df)
        record = batch.iloc[0].to_dict()

        # Build inference payload
        inference_params = {
            MAX_NEW_TOKENS: self._args.teacher_model_max_new_tokens,
            TEMPERATURE: self._args.teacher_model_temperature,
            TOP_P: self._args.teacher_model_top_p,
            **record,
        }

        headers = self._get_inference_request_headers()
        url = self._args.teacher_model_endpoint_url
        url = url if VLLM_CHAT_SCORE_PATH in url else f"{url}{VLLM_CHAT_SCORE_PATH}"
        logger.info(f"Model endpoint: {url}")
        response = requests.post(
            url=url, headers=headers, data=json.dumps(inference_params), timeout=180
        )
        response.raise_for_status()

    def _validate_inference_parameters(self):
        """Validate all body parameters passed as part of inference."""
        validate_model_temperature(self._args.teacher_model_temperature)
        validate_model_top_p(self._args.teacher_model_top_p)
        validate_model_presence_penalty(self._args.teacher_model_presence_penalty)
        validate_model_frequency_penalty(self._args.teacher_model_frequency_penalty)

        validate_request_batch_size(self._args.request_batch_size)
        validate_min_endpoint_success_ratio(self._args.min_endpoint_success_ratio)

    def _validate_number_of_records(self, size: int):
        """Validate number of records in the dataset."""
        task_type = self._args.data_generation_task_type
        min_records = MIN_RECORDS_FOR_FT if task_type != DataGenerationTaskType.MATH else MATH_MIN_RECORDS_FOR_FT
        if size < min_records:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "Number of records in the dataset are less than the minimum required for fine-tuning."
                        f" Minimum records required for task {task_type}: {min_records}, but got {size}."
                    ),
                )
            )

    def _validate_record_for_type_conversation(self, record: list) -> str:
        if self._args.data_generation_task_type != DataGenerationTaskType.CONVERSATION:
            return

        if self._get_cot_status():
            return f"Chain of thought is not supported for task type {DataGenerationTaskType.CONVERSATION}"

        if self._get_max_len_summary():
            return f"Max length summary is not supported for task type {DataGenerationTaskType.CONVERSATION}"

        if len(record) < 3:
            return f"Dataset is not matching expected schema for task type {DataGenerationTaskType.CONVERSATION}. \
                Expected format: [system, user, assistant]"

    def _validate_record_for_type_NLI(self, record: list) -> str:
        if self._args.data_generation_task_type != DataGenerationTaskType.NLI:
            return

        if self._get_cod_status():
            return f"Chain of density is not supported for task type {DataGenerationTaskType.NLI}"

        if self._get_max_len_summary():
            return f"Max length summary is not supported for task type {DataGenerationTaskType.NLI}"

        if len(record) > 2:
            return f"Chat cannot be of type multi-turn for task type {DataGenerationTaskType.NLI}. \
                Expected format: [system, user]"

    def _validate_record_for_type_NLU_QA(self, record: list) -> str:
        if (
            self._args.data_generation_task_type
            != DataGenerationTaskType.NLU_QUESTION_ANSWERING
        ):
            return

        if self._get_cod_status():
            return f"Chain of density is not supported for task type {DataGenerationTaskType.NLU_QUESTION_ANSWERING}"

        if self._get_max_len_summary():
            return f"Max length summary is not supported for task type {DataGenerationTaskType.NLU_QUESTION_ANSWERING}"

        if len(record) > 2:
            return f"Chat cannot be of type multi-turn for task type {DataGenerationTaskType.NLU_QUESTION_ANSWERING} \
                Expected format: [system, user]"

    def _validate_record_for_type_MATH(self, record: list) -> str:
        if self._args.data_generation_task_type != DataGenerationTaskType.MATH:
            return

        if self._get_max_len_summary():
            return f"Max length summary is not supported for task type {DataGenerationTaskType.MATH}"

        if len(record) > 2:
            return f"Chat cannot be of type multi-turn for task type {DataGenerationTaskType.MATH} \
                Expected format: [system, user]"

    def _validate_record_for_type_summarization(self, record: list) -> str:
        if (
            self._args.data_generation_task_type
            != DataGenerationTaskType.SUMMARIZATION
        ):
            return

        if self._get_cot_status():
            return f"Chain of thought is not supported for task type {DataGenerationTaskType.SUMMARIZATION}"

        if len(record) > 2:
            return f"Chat cannot be of type multi-turn for task type {DataGenerationTaskType.SUMMARIZATION} \
                Expected format: [system, user]"

    def _validate_record_by_task(self, record: list) -> dict:
        """
        Validate record in a dataset against the data generation task type.

        Returns a dictionary containing exception if any validation error is found.

        Args:
            record (list): Sequence of messages
        """
        validation_methods = [
            self._validate_record_for_type_NLI,
            self._validate_record_for_type_conversation,
            self._validate_record_for_type_NLU_QA,
            self._validate_record_for_type_MATH,
            self._validate_record_for_type_summarization,
        ]

        for method in validation_methods:
            err = method(record=record)
            if err:
                return {"exception": err}

    def _validate_message(self, id: int, message: dict) -> dict:
        """
        Validate individual message in the dataset.

        Returns dictionary containing exception, if any validation error is found.

        Args:
            id (int): id of the message in sequence of messages.
            message (dict): Message object in sequence of messages.
        """
        allowed_roles = ["system", "user", "assistant"]
        if "role" not in message:
            return f"Message at index {id} is missing 'role'."

        if message["role"] not in allowed_roles:
            return f"Invalid 'role' at index {id}."

        if "content" not in message:
            return f"Message at index {id} is missing 'content'."

    def _validate_record_content(self, record: list) -> dict:
        """
        Validate content of a record and ensures messages are in the expected format.

        Currently functional only for task type `CONVERSATION`, `SUMMARIZATION`, `NLI` & `NLU`.
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
                    return {"exception": err}

                expected_role = expected_roles[(id - 1) % 2]
                if message.get("role") != expected_role:
                    return {
                        "exception": f"Role at index {id} should be {expected_role}."
                    }

            task_type = self._args.data_generation_task_type
            if task_type == DataGenerationTaskType.CONVERSATION and (
                len(record[1:]) % 2 != 0
            ):
                return {
                    "exception": "There is an incomplete pair of 'user' and 'assistant' messages."
                }

        except Exception as e:
            return {"exception": e}

    def _validate_dataset_record(self, record: list) -> str:
        """Validate a record in the dataset. Returns the validation error if found.

        Args:
            record (list): Sequence of messages
        """
        if not record:
            return "Chat cannot be empty."

        err = self._validate_record_by_task(record=record)
        if err and ("exception" in err):
            return err["exception"]

        err = self._validate_record_content(record=record)
        if err and ("exception" in err):
            return err["exception"]

    def _validate_dataset(self, file_path: str):
        """Validate training/validation dataset passed to the data-generation component.

        Args:
            file_path (str): Path to the dataset

        Raises:
            ACFTUserError: If a known validation error is caught
        """
        df = self._get_dataframe(file_path=file_path)
        total_rows = 0
        for batch in df:
            total_rows += len(batch)
            for idx, row in batch.iterrows():
                record = row.iloc[0]
                err = self._validate_dataset_record(record=record)
                if err:
                    raise ACFTValidationException._with_error(
                        AzureMLError.create(
                            ACFTUserError,
                            pii_safe_message=(
                                f"Error validating dataset record, context({idx}): {err}"
                            ),
                        )
                    )

        self._validate_number_of_records(size=total_rows)

    def _validate_data_generation_inputs(self):
        """Validate all input flags to the data-generation component.

        Sequentially performs a set of validations, each dependent on the previous validation.
        1. Validate training/validation file paths and ensure files exist.
        2. Validate teacher model endpoint arguments are passed for inference, and
        authenticity of the endpoint.
        3. Validate that the passed inference parameters are within limits.
        4. Validate integrity of datasets.
        5. Validate a single inference call to the teacher model.
        """
        with log_activity(
            logger=logger, activity_name=TelemetryConstants.VALIDATE_FILE_PATH
        ):
            files = [self._args.train_file_path, self._args.validation_file_path]
            validate_file_paths_with_supported_formats(file_paths=files)
            validate_file_exists(file_paths=files)

        with log_activity(
            logger=logger,
            activity_name=TelemetryConstants.VALIDATE_TEACHER_MODEL_ENDPOINT,
        ):
            self._validate_model_endpoint_args()
            self._validate_model_endpoint()

        with log_activity(
            logger=logger,
            activity_name=TelemetryConstants.VALIDATE_INFERENCE_PARAMETERS,
        ):
            self._validate_inference_parameters()

        with log_activity(
            logger=logger, activity_name=TelemetryConstants.VALIDATE_TRAINING_DATA
        ):
            self._validate_dataset(self._args.train_file_path)

        if self._args.validation_file_path:
            with log_activity(
                logger=logger, activity_name=TelemetryConstants.VALIDATE_VALIDATION_DATA
            ):
                self._validate_dataset(self._args.validation_file_path)

        with log_activity(
            logger=logger, activity_name=TelemetryConstants.VALIDATE_MODEL_INFERENCE
        ):
            self._validate_model_inference()


@swallow_all_exceptions(time_delay=5)
def main():
    """Run validation."""
    # Get data generation component input parameters.
    parser = get_parser()
    parser.add_argument("--validation_info", required=True, help="Validation status")
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type="DistillationPipelineValidation",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    with log_activity(logger=logger, activity_name=TelemetryConstants.VALIDATOR):
        PipelineInputsValidator(args=args)

    if args.validation_info:
        with open(args.validation_info, "w") as f:
            f.write(json.dumps({"validation_status": "ok"}))


if __name__ == "__main__":
    main()

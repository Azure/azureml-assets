# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for FTaaS data import component."""

import json
import logging
import pandas as pd

import argparse
import requests
from argparse import Namespace
from requests import Response
from pathlib import Path
from typing import List, Optional

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml._common._error_definition.azureml_error import AzureMLError

from concurrent.futures import ThreadPoolExecutor, as_completed

from common.constants import (
    COMPONENT_NAME,
    COT_SYSTEM_PROMPT,
    DEFAULT_REQUEST_BATCH_SIZE,
    DEFAULT_SUCCESS_RATIO,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
    MAX_NEW_TOKENS,
    MAX_BATCH_SIZE,
    TEMPERATURE,
    TOP_P,
    STOP_TOKEN,
    SUPPORTED_FILE_FORMATS,
    VLLM_CHAT_SCORE_PATH,
    DataGenerationTaskType
)

from common.utils import (
    get_workspace_mlclient,
    get_endpoint_details,
    validate_teacher_model_details,
    retry,
)


logger = get_logger_app("azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import")


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model selector for hugging face models", allow_abbrev=False)

    # File I/O
    parser.add_argument(
        "--train_file_path",
        type=str,
        help="Input train file path",
    )

    parser.add_argument(
        "--validation_file_path",
        default=None,
        type=str,
        help="Input validation file path",
    )

    parser.add_argument(
        "--generated_train_file_path",
        type=Path,
        default=None,
        help="file to save the generated training data",
    )

    parser.add_argument(
        "--generated_validation_file_path",
        type=Path,
        default=None,
        help="file to save the generated validation data",
    )

    # add optional data-generator params
    parser.add_argument(
        "--teacher_model_endpoint_name",
        type=str,
        required=False,
        help="Teacher model endpoint name",
    )
    parser.add_argument(
        "--teacher_model_endpoint_url",
        type=str,
        required=False,
        help="Teacher model endpoint URL",
    )
    parser.add_argument(
        "--teacher_model_endpoint_key",
        type=str,
        required=False,
        help="Teacher model endpoint key",
    )
    parser.add_argument(
        "--teacher_model_max_new_tokens",
        type=int,
        required=False,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Teacher model max_tokens parameter",
    )
    parser.add_argument(
        "--teacher_model_temperature",
        type=float,
        required=False,
        default=DEFAULT_TEMPERATURE,
        help="Teacher model temperature parameter",
    )
    parser.add_argument(
        "--teacher_model_top_p",
        type=float,
        required=False,
        default=DEFAULT_TOP_P,
        help="Teacher model top-p parameter"
    )
    parser.add_argument(
        "--teacher_model_frequency_penalty",
        type=float,
        required=False,
        help="Teacher model frequency parameter"
    )
    parser.add_argument(
        "--teacher_model_presence_penalty",
        type=float,
        required=False,
        help="Teacher model presense penalty"
    )
    parser.add_argument(
        "--teacher_model_stop",
        type=str,
        required=False,
        help="Teacher model stop "
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=DEFAULT_REQUEST_BATCH_SIZE,
        required=False,
        help="No of data records to process at a time.",
    )
    parser.add_argument(
        "--min_endpoint_success_ratio",
        type=float,
        required=False,
        default=DEFAULT_SUCCESS_RATIO,
        help=(
            f"The minimum value of "
            "(successful_requests / total_requests) required for classifying inference as successful. "
            "If (successful_requests / total_requests) < min_endpoint_success_ratio, "
            "the experiment will be marked as failed. "
            f"By default it is {DEFAULT_SUCCESS_RATIO}. "
            "(0 means all requests are allowed to fail while 1 means no request should fail.)"
        )
    )

    parser.add_argument(
        "--enable_chain_of_thought",
        type=str,
        required=False,
        default="false",
        help="This enables Chain of Thought"
    )

    parser.add_argument(
        "--data_generation_task_type",
        type=str,
        required=True,
        help="""Data generation task type. Supported values are:
            1. NLI: Generate Natural Language Inference data
            2. CONVERSATION: Generate conversational data (multi/single turn)
            3. NLU_QA: Generate Natural Language Understanding data for Question Answering data
            """,
        choices=[v.value for v in DataGenerationTaskType]
    )

    return parser


@retry(3)
def _invoke_endpoint(url: str, key: str, data: dict) -> Response:
    """Invoke endpoint with payload data.

    Args:
        url (str): Endpoint URL
        key (str): Endpoint key
        data dict): Payload dictionary

    Returns:
        Response: Response from invocation
    """
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    response = requests.post(url, headers=request_headers, data=json.dumps(data))
    return response


def _validate_file_paths_with_supported_formats(file_paths: List[Optional[str]]):
    """Check if the file path is in the list of supported formats."""
    for file_path in file_paths:
        if file_path:
            file_suffix = Path(file_path).suffix.lower()
            file_ext = file_suffix.split('?')[0]
        if file_ext and file_ext not in SUPPORTED_FILE_FORMATS:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"{file_path} is not in list of supported file formats. "
                        f"Supported file formats: {SUPPORTED_FILE_FORMATS}"
                    )
                )
            )


def generate_synthetic_data(
    teacher_model_endpoint_url: str,
    teacher_model_endpoint_key: str,
    inference_params: dict,
    request_batch_size: int,
    min_endpoint_success_ratio: float,
    enable_cot: bool,
    generated_train_file_path: Path,
    generated_validation_file_path: Path,
    train_file_path: Path,
    data_generation_task_type: str,
    validation_file_path: Path = None
):
    """Generate and save synthentic data under output_dataset.

    Args:
        teacher_model_endpoint_url (str): Teacher model endpoint URL
        teacher_model_endpoint_key (str): Teacher model endpoint key
        inference_params (dict): Inference params to hit endpoint with
        request_batch_size (int): Input batch size for processing rows in train and validation dataset
        min_endpoint_success_ratio (float): Minimum success ratio below which run will be considered a failure
        enable_cot (bool): Enable Chain of Thought processing
        output_dataset (Path): Path to output directory
        train_file_path (Path): Train JSONL file path
        validation_file_path (Path, optional): Validation JSONL file path. Defaults to None.
    """
    def process_request(idx: str, enable_cot: bool, data: dict, url: str, endpoint_key: str) -> dict:
        """Process a single request.

        Args:
            idx (str): Row index in Input data.
            enable_cot (bool): If CoT is enabled
            data (dict): Payload dict
            url (str): Endpoint URL
            endpoint_key (str): key to authenticate endpoint request

        Returns:
            dict: result dictionary
        """
        try:
            response: Response = _invoke_endpoint(url=url, key=endpoint_key, data=data)
            response_data = response.json()

            # use jsonpath or regex to capture prediction result
            prediction_result = (
                None if response.status_code != 200
                # response content should be structured as below for a successful vllm response
                else response_data['choices'][0]["message"]["content"].strip()
            )

            if enable_cot:
                # Try loading JSON answer and filter 'answer_choice'
                # if JSON loading fails, exception will be caught
                # And this specific row would not be part of generated data
                prediction_result = json.loads(prediction_result)['answer_choice']

            return {
                "idx": idx,
                "status_code": response.status_code,
                "text": prediction_result,
                "exception": None,
            }
        except Exception as e:
            logger.error(f"idx: {idx}. exception: {e}")
            return {
                "idx": idx,
                "status_code": None,
                "text": None,
                "exception": e,
            }

    def process_conversational_request(idx: str, data: dict, url: str, endpoint_key: str):
        """Process a single conversational request.

        Args:
            idx (str): Row index in Input data.
            data (dict): Payload dict
            url (str): Endpoint URL
            endpoint_key (str): key to authenticate endpoint request

        Returns:
            dict: result dictionary
        """
        try:
            logger.info(f"request_data: {repr(data)}")
            #  Basic validation for the input data
            messages = data.pop("messages", [])
            if not messages:  # empty messages
                return {
                    "idx": idx,
                    "status_code": None,
                    "messages": [],
                    "exception": "Empty messages"
                }
            first_message = messages[0]
            if first_message['role'] != 'system':
                logger.warning(f"First message should be system, but got {first_message['role']}")
                return {"idx": idx,
                        "status_code": None,
                        "messages": [],
                        "exception": ("Incorrect format.\n"
                                      f"First message should be system, but got {first_message['role']}"),
                        }
            for message in messages[1:]:
                role = message['role']
                if role not in ('assistant', 'user'):
                    logger.warning(f"role should be system or user, but got {role}")
                    return {"idx": idx,
                            "status_code": None,
                            "messages": [],
                            "exception": f"Incorrect format.\nRole should be assistant or user, but got {role}"
                            }

            synthetic_responses = []
            for message in messages:
                role = message['role']
                if role in ('system', 'user'):
                    synthetic_responses.append(message)
                else:
                    data_with_inference_parameters = {"messages": synthetic_responses}
                    for key, value in data.items():
                        data_with_inference_parameters[key] = value
                    # replace the assistant content from the model
                    response: Response = _invoke_endpoint(url=url, key=endpoint_key,
                                                          data=data_with_inference_parameters)
                    if response.status_code != 200:
                        break
                    response_data = response.json()
                    prediction_result = (
                        None if response.status_code != 200
                        # response content should be structured as below for a successful vllm response
                        else response_data['choices'][0]["message"]["content"].strip()
                    )
                    synthetic_responses.append({'role': 'assistant', 'content': prediction_result})
            return {
                "idx": idx,
                "status_code": response.status_code,
                "messages": synthetic_responses,
                "exception": (f"Not able to generate synthetic response for all turns for idx: {idx}"
                              if response.status_code != 200
                              else
                              None),
            }
        except Exception as e:
            logger.error(f"idx: {idx}. exception: {e}")
            return {
                "idx": idx,
                "status_code": None,
                "messages": [],
                "exception": e,
            }

    def replace_cot_system_message(messages: List[dict]) -> List[dict]:
        # Replace the system message without changing the original messages list
        cot_system_message = {'role': 'system', 'content': COT_SYSTEM_PROMPT}
        return [(cot_system_message if message['role'] == 'system' else message) for message in messages]

    def batch_process_conversation_data(input_file_path: Path, output_file_path: Path, batch_size: int) -> None:
        """Batch process data and do a bulk request to teacher model endpoint.

        Args:
            input_file_path (Path): Input data file path
            output_file_path (Path): Path to output directory
            batch_size (int): Input batch size for processing rows in train and validation dataset

        Raises:
            Exception: if success ratio is less than min_endpoint_success_ratio
        """
        train_df = pd.read_json(input_file_path, lines=True, chunksize=batch_size)
        total_rows = 0
        error_count = 0
        output_data = []
        error_map = {}
        ERROR = "error"

        for batch in train_df:
            total_rows += len(batch)
            futures = []

            with ThreadPoolExecutor() as executor:
                for idx, row in batch.iterrows():
                    messages = row.iloc[0]
                    request_data = {
                        "messages": messages,
                        **inference_params,
                    }
                    futures.append(
                        executor.submit(
                            process_conversational_request,
                            idx,
                            request_data,
                            teacher_model_endpoint_url,
                            teacher_model_endpoint_key
                        )
                    )

            # wait for results to complete
            future_results = {
                result["idx"]: result
                for result in [future.result() for future in as_completed(futures)]
            }

            idx = 0
            for idx, row in batch.iterrows():
                future_result = future_results.get(idx)
                if future_result is None:
                    logger.error(f"row {idx} not found in future_results")
                    error_map[ERROR] = error_map.get(ERROR, 0) + 1
                elif future_result['exception']:
                    logger.error(f"row {idx} failed with exception: {future_result['exception']}")
                    error_map[ERROR] = error_map.get(ERROR, 0) + 1
                elif future_result['status_code'] != 200:
                    logger.warning(f"row {idx} request status_code: {future_result['status_code']} != 200")
                    error_map[future_result['status_code']] = error_map.get(future_result['status_code'], 0) + 1
                else:
                    output_data.append({"messages": future_result['messages']})
            Path(output_file_path.parent).mkdir(exist_ok=True, parents=True)
            with open(output_file_path, 'w') as f:
                for entry in output_data:
                    f.write(json.dumps(entry) + '\n')

        if error_map:
            logger.info("Error summary. With key denoting non-200 status code or some other error.")
            for k, v in error_map.items():
                error_count += v
                logger.warning(f"{k} => {v}")

        success_ratio = float(total_rows - error_count) / total_rows
        logger.info(f"Success rate was {success_ratio} for {input_file_path}")
        if success_ratio < min_endpoint_success_ratio:
            msg = f"Success ratio for dataset {input_file_path}: {success_ratio} < {min_endpoint_success_ratio}."
            raise Exception(msg)

    def batch_process_data(input_file_path: Path, output_file_path: Path, batch_size: int) -> None:
        """Batch process data and do a bulk request to teacher model endpoint.

        Args:
            input_file_path (Path): Input data file path
            output_file_path (Path): Path to output directory
            batch_size (int): Input batch size for processing rows in train and validation dataset

        Raises:
            Exception: if success ratio is less than min_endpoint_success_ratio
        """
        train_df = pd.read_json(input_file_path, lines=True, chunksize=batch_size)
        total_rows = 0
        error_count = 0
        output_data = []
        error_map = {}
        ERROR = "error"

        for batch in train_df:
            total_rows += len(batch)
            futures = []

            with ThreadPoolExecutor() as executor:
                for idx, row in batch.iterrows():
                    messages = row.iloc[0]
                    messages = replace_cot_system_message(messages) if enable_cot else messages
                    request_data = {
                        "messages": messages,
                        **inference_params,
                    }

                    futures.append(
                        executor.submit(
                            process_request,
                            idx,
                            enable_cot,
                            request_data,
                            teacher_model_endpoint_url,
                            teacher_model_endpoint_key
                        )
                    )

            # wait for results to complete
            future_results = {
                result["idx"]: result
                for result in [future.result() for future in as_completed(futures)]
            }

            idx = 0
            for idx, row in batch.iterrows():
                future_result = future_results.get(idx)
                if future_result['exception']:
                    logger.error(f"row {idx} failed with exception: {future_result['exception']}")
                    error_map[ERROR] = error_map.get(ERROR, 0) + 1
                elif future_result['status_code'] != 200:
                    logger.warning(f"row {idx} request status_code: {future_result['status_code']} != 200")
                    error_map[future_result['status_code']] = error_map.get(future_result['status_code'], 0) + 1
                else:
                    new_row = row.copy().iloc[0]
                    answer = future_result['text']

                    new_row.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )
                    output_data.append({"messages": new_row})

            Path(output_file_path.parent).mkdir(exist_ok=True, parents=True)
            with open(output_file_path, 'w') as f:
                for entry in output_data:
                    f.write(json.dumps(entry) + '\n')

        if error_map:
            logger.info("Error summary. With key donating non-200 status code or some other error.")
            for k, v in error_map.items():
                error_count += v
                logger.warning(f"{k} => {v}")

        success_ratio = float(total_rows - error_count) / total_rows
        logger.info(f"Success rate was {success_ratio} for {input_file_path}")
        if success_ratio < min_endpoint_success_ratio:
            msg = f"Success ratio for dataset {input_file_path}: {success_ratio} < {min_endpoint_success_ratio}."
            raise Exception(msg)
    logger.info("Processing train file")

    if data_generation_task_type == DataGenerationTaskType.CONVERSATION:
        batch_process_conversation_data(train_file_path, generated_train_file_path, request_batch_size)
    else:
        batch_process_data(train_file_path, generated_train_file_path, request_batch_size)

    logger.info("Data generated and saved for train file")

    if validation_file_path:
        logger.info("Processing validation file")
        if data_generation_task_type == DataGenerationTaskType.CONVERSATION:
            batch_process_conversation_data(validation_file_path, generated_validation_file_path, request_batch_size)
        else:
            batch_process_data(validation_file_path, generated_validation_file_path, request_batch_size)
        logger.info("Data generated and saved for validation file")


def data_import(args: Namespace):
    """Copy the user data to output dir."""
    train_file_path = args.train_file_path
    validation_file_path = args.validation_file_path
    generated_train_file_path = args.generated_train_file_path
    generated_validation_file_path = args.generated_validation_file_path

    # add optional data-generator params
    teacher_model_endpoint_name = args.teacher_model_endpoint_name
    teacher_model_endpoint_url = args.teacher_model_endpoint_url
    teacher_model_endpoint_key = args.teacher_model_endpoint_key
    teacher_model_max_new_tokens = args.teacher_model_max_new_tokens
    teacher_model_temperature = args.teacher_model_temperature
    teacher_model_top_p = args.teacher_model_top_p
    teacher_model_frequency_penalty = args.teacher_model_frequency_penalty
    teacher_model_presence_penalty = args.teacher_model_presence_penalty
    teacher_model_stop = args.teacher_model_stop
    request_batch_size = args.request_batch_size
    min_endpoint_success_ratio = args.min_endpoint_success_ratio
    enable_cot_str = args.enable_chain_of_thought
    data_generation_task_type = args.data_generation_task_type

    # validate file formats
    _validate_file_paths_with_supported_formats([args.train_file_path, args.validation_file_path])
    logger.info("File format validation successful.")

    enable_cot = True if enable_cot_str.lower() == "true" else False
    mlclient_ws = get_workspace_mlclient()
    if not mlclient_ws:
        raise Exception("Could not create MLClient for current workspace")

    if teacher_model_endpoint_name:
        endpoint_details = get_endpoint_details(mlclient_ws, teacher_model_endpoint_name)
        teacher_model_endpoint_key = endpoint_details.get_endpoint_key()
        teacher_model_endpoint_url = endpoint_details.get_endpoint_url()
        teacher_model_asset_id = endpoint_details.get_deployed_model_id()
        validate_teacher_model_details(teacher_model_asset_id)

    if not teacher_model_endpoint_url:
        raise Exception("Endpoint URL is a requried parameter for data generation")

    if not teacher_model_endpoint_key:
        raise Exception("Endpoint key is a requried parameter for data generation")

    if teacher_model_top_p < 0 or teacher_model_top_p > 1:
        raise Exception(
            f"Invalid teacher_model_top_p. Value should be 0<=val<=1, but it is {teacher_model_top_p}")
    if teacher_model_temperature < 0 or teacher_model_temperature > 1:
        raise Exception(
            f"Invalid teacher_model_temperature. Value should be 0<=val<=1, but it is {teacher_model_temperature}")
    if min_endpoint_success_ratio < 0 or min_endpoint_success_ratio > 1:
        raise Exception(
            f"Invalid min_endpoint_success_ratio. Value should be 0<=val<=1, but it is {min_endpoint_success_ratio}")

    if request_batch_size <= 0 or request_batch_size > MAX_BATCH_SIZE:
        raise Exception(
            f"Invalid request_batch_size. Value should be 0<=val<={MAX_BATCH_SIZE}, but it is {request_batch_size}")

    inference_params = {
        MAX_NEW_TOKENS: teacher_model_max_new_tokens,
        TEMPERATURE: teacher_model_temperature,
        TOP_P: teacher_model_top_p
    }

    if teacher_model_frequency_penalty:
        inference_params[FREQUENCY_PENALTY] = teacher_model_frequency_penalty

    if teacher_model_presence_penalty:
        inference_params[PRESENCE_PENALTY] = teacher_model_presence_penalty

    if teacher_model_stop:
        inference_params[STOP_TOKEN] = teacher_model_stop

    if VLLM_CHAT_SCORE_PATH not in teacher_model_endpoint_url:
        teacher_model_endpoint_url += VLLM_CHAT_SCORE_PATH

    logger.info(f"Teacher Endpoint : {teacher_model_endpoint_url}")

    logger.info("Running data generation")
    generate_synthetic_data(
        teacher_model_endpoint_url=teacher_model_endpoint_url,
        teacher_model_endpoint_key=teacher_model_endpoint_key,
        inference_params=inference_params,
        request_batch_size=request_batch_size,
        min_endpoint_success_ratio=min_endpoint_success_ratio,
        enable_cot=enable_cot,
        generated_train_file_path=generated_train_file_path,
        generated_validation_file_path=generated_validation_file_path,
        train_file_path=train_file_path,
        data_generation_task_type=data_generation_task_type,
        validation_file_path=validation_file_path,
    )


@swallow_all_exceptions(time_delay=5)
def main():
    """Parse args and import model."""
    parser = get_parser()
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type="ChatCompletion",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    logger.info(args)
    data_import(args)


if __name__ == "__main__":
    main()

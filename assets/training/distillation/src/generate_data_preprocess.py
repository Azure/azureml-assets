# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for FTaaS data import component."""

import json
import logging
import argparse
from argparse import Namespace
from pathlib import Path
import os
import uuid
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import (
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
)
from azureml.acft.common_components import (
    get_logger_app,
    set_logging_parameters,
    LoggingLiterals,
)
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.telemetry.activity import log_activity
from azure.ai.ml.entities import ServerlessConnection

from common.io import read_jsonl_files

from mltable import from_json_lines_files

from common.constants import (
    COMPONENT_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    STOP_TOKEN,
    VLLM_CHAT_SCORE_PATH,
    DataGenerationTaskType,
    HashField,
    TelemetryConstants,
    SystemPrompt,
    PayloadField,
    DEFAULT_MAX_LEN_SUMMARY,
)

from common.utils import (
    get_workspace_mlclient,
    get_endpoint_details,
    get_hash_value,
    validate_teacher_model_details,
)

from common.validation import validate_file_paths_with_supported_formats

logger = get_logger_app(
    "azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import"
)


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(
        description="Model selector for hugging face models", allow_abbrev=False
    )

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
        "--teacher_model_endpoint_key",
        type=str,
        required=False,
        help="Teacher model endpoint key",
    )

    parser.add_argument(
        "--teacher_model_endpoint_url",
        type=str,
        required=True,
        help="Teacher model endpoint URL",
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
        help="Teacher model top-p parameter",
    )
    parser.add_argument(
        "--teacher_model_frequency_penalty",
        type=float,
        required=False,
        help="Teacher model frequency parameter",
    )
    parser.add_argument(
        "--teacher_model_presence_penalty",
        type=float,
        required=False,
        help="Teacher model presense penalty",
    )
    parser.add_argument(
        "--teacher_model_stop", type=str, required=False, help="Teacher model stop "
    )

    parser.add_argument(
        "--enable_chain_of_thought",
        type=str,
        required=False,
        default="false",
        help="This enables Chain of Thought",
    )

    parser.add_argument(
        "--enable_chain_of_density",
        type=str,
        required=False,
        default="false",
        help="This enables Chain of Density for Summarization",
    )

    parser.add_argument(
        "--max_len_summary",
        type=int,
        required=False,
        default=DEFAULT_MAX_LEN_SUMMARY,
        help="Maximum word count for text summarization ",
    )

    parser.add_argument(
        "--data_generation_task_type",
        type=str,
        required=True,
        help="""Data generation task type. Supported values are:
            1. NLI: Generate Natural Language Inference data
            2. CONVERSATION: Generate conversational data (multi/single turn)
            3. NLU_QA: Generate Natural Language Understanding data for Question Answering data
            4. MATH: Generate Math data for numerical responses
            5. SUMMARIZATION: Generate Text Summary for Article
            """,
        choices=[v.value for v in DataGenerationTaskType],
    )

    parser.add_argument(
        "--generated_train_payload_path",
        type=str,
        help="file to save the generated training payload data",
    )

    parser.add_argument(
        "--generated_validation_payload_path",
        type=str,
        help="file to save the generated validation payload data",
    )

    parser.add_argument(
        "--hash_train_data",
        type=str,
        required=True,
        help="Path tho the jsonl file where the hash for each payload will be dumped.",
    )

    parser.add_argument(
        "--hash_validation_data",
        type=str,
        required=True,
        help="Path tho the jsonl file where the hash for each payload will be dumped.",
    )

    parser.add_argument(
        "--batch_config_connection",
        type=str,
        required=True,
        help="",
    )

    return parser


def preprocess_data(
    inference_params: dict,
    enable_cot: bool,
    enable_cod: bool,
    max_len_summary: int,
    data_generation_task_type: str,
    generated_train_payload_path: str,
    generated_validation_payload_path: str,
    hash_train_data: str,
    hash_validation_data: str,
    train_file_path: Path,
    validation_file_path: Path = None,
):
    """Generate and save synthentic data under output_dataset.

    Args:
        inference_params (dict): Inference params to hit endpoint with
        enable_cot (bool): Enable Chain of Thought processing
        enable_cod (bool): Enable Chain of Density processing for text summarization task
        max_len_summary (int): Maximum word count for text summarization
        data_generation_task_type (str): Data generation task type
        generated_train_payload_path (str): Path to save the generated training payload data
        generated_validation_payload_path (str): Path to save the generated validation payload data
        hash_train_data (str): Path to the jsonl file where the hash for each payload will be dumped.
        hash_validation_data (str): Path to the jsonl file where the hash for each payload will be dumped.
        train_file_path (Path): Train JSONL file path
        validation_file_path (Path, optional): Validation JSONL file path. Defaults to None.
    """

    def process_system_prompt(message: dict) -> dict:
        """Update the system prompt depending on the task type and the flag enable_cot.

        The original message unchanged if enable_cot is False or task type is conversation.

        Args:
            message (dict): System message

        Returns:
            message (dict): System message with updated content
        """
        if (
            enable_cot
            and data_generation_task_type != DataGenerationTaskType.CONVERSATION
        ):
            cot_prompt = SystemPrompt.get_cot_prompt(data_generation_task_type)
            cot_system_message = {"role": "system", "content": cot_prompt}
            return cot_system_message
        elif (
            enable_cod
            and data_generation_task_type == DataGenerationTaskType.SUMMARIZATION
        ):
            cod_prompt = SystemPrompt.get_cod_prompt(max_len_summary)
            cod_system_message = {"role": "system", "content": cod_prompt}
            return cod_system_message
        else:
            return message

    def pre_process_data(
        input_file_path: Path,
        output_file_path: Path,
        hash_file_path: str,
    ) -> None:
        """Batch process data and do a bulk request to teacher model endpoint.

        Args:
            input_file_path (Path): Input data file path
            output_file_path (Path): Path to output directory
            hash_file_path (str): Path to the jsonl file where the hash for each payload will be dumped.
        Raises:
            Exception: if success ratio is less than min_endpoint_success_ratio
        """
        input_data = read_jsonl_files(input_file_path)
        output_data = []
        output_hash_data = []
        try:
            for idx, record in enumerate(input_data):
                #  Basic validation for the input data
                messages = record.pop("messages", [])
                if not messages:  # empty messages
                    logger.error(f"Failed with exception:{idx} Empty messages")
                    return
                first_message = messages[0]
                if first_message["role"] != PayloadField.SYSTEM:
                    logger.error(
                        f"row {idx} failed with exception: First message should be system, "
                        f"but got {first_message['role']}"
                    )
                for message in messages[1:]:
                    role = message["role"]
                    if role not in ("assistant", "user"):
                        logger.error(
                            f"row {idx} failed with exception: role should be system or user, but got {role}"
                        )
                inference_data = []
                system_message = {}
                for message in messages:
                    role = message["role"]
                    if role == PayloadField.SYSTEM:
                        system_message[PayloadField.SYSTEM] = message
                        inference_data.append(process_system_prompt(message))
                    elif role == PayloadField.USER:
                        inference_data.append(message)
                        hash_data = {
                            HashField.HASH: get_hash_value(message),
                            **system_message,
                        }
                output_data.append(
                    {
                        "messages": inference_data,
                        **inference_params,
                    }
                )
                output_hash_data.append(hash_data)
        except Exception as e:
            logger.error(f"idx: {idx}. exception: {e}")
        payload_jsonl_path = os.path.join(output_file_path, "payload.jsonl")
        logger.info("payload_jsonl_path: %s", payload_jsonl_path)
        with open(payload_jsonl_path, "w") as payload_file:
            for entry in output_data:
                payload_file.write(json.dumps(entry) + "\n")
        logger.info("hash_file_path: %s", hash_file_path)
        with open(hash_file_path, "w") as hash_file:
            for entry in output_hash_data:
                hash_file.write(json.dumps(entry) + "\n")

        output_file_path = str(output_file_path)
        mltable = from_json_lines_files(paths=[{"file": payload_jsonl_path}])
        logger.info("output_file_path type before saving: %s", output_file_path)
        mltable.save(output_file_path)

    with log_activity(
        logger=logger, activity_name=TelemetryConstants.PRE_PROCESS_TRAINING_DATA
    ):
        logger.info("PreProcessing train file")
        pre_process_data(train_file_path, generated_train_payload_path, hash_train_data)
        logger.info("Data generated and saved for train file")

    if validation_file_path:
        with log_activity(
            logger=logger,
            activity_name=TelemetryConstants.PRE_PROCESS_VALIDATION_DATA,
        ):
            logger.info("PreProcessing validation file")
            pre_process_data(
                validation_file_path,
                generated_validation_payload_path,
                hash_validation_data,
            )
            logger.info("Data generated and saved for validation file")
    else:
        hash_validation_data = Path(hash_validation_data)
        Path(hash_validation_data.parent).mkdir(exist_ok=True, parents=True)
        # create an empty file if validation file is not provided
        open(hash_validation_data, "w").close()


def data_import(args: Namespace):
    """Copy the user data to output dir."""
    train_file_path = args.train_file_path
    validation_file_path = args.validation_file_path
    generated_train_payload_path = args.generated_train_payload_path
    generated_validation_payload_path = args.generated_validation_payload_path
    teacher_model_endpoint_name = args.teacher_model_endpoint_name
    teacher_model_endpoint_url = args.teacher_model_endpoint_url
    teacher_model_endpoint_key = args.teacher_model_endpoint_key
    # add optional data-generator params
    teacher_model_max_new_tokens = args.teacher_model_max_new_tokens
    teacher_model_temperature = args.teacher_model_temperature
    teacher_model_top_p = args.teacher_model_top_p
    teacher_model_frequency_penalty = args.teacher_model_frequency_penalty
    teacher_model_presence_penalty = args.teacher_model_presence_penalty
    teacher_model_stop = args.teacher_model_stop
    enable_cot_str = args.enable_chain_of_thought
    enable_cod_str = args.enable_chain_of_density
    max_len_summary = args.max_len_summary
    data_generation_task_type = args.data_generation_task_type
    hash_train_data = args.hash_train_data
    hash_validation_data = args.hash_validation_data
    batch_config_connection = args.batch_config_connection

    # validate file formats
    validate_file_paths_with_supported_formats(
        [args.train_file_path, args.validation_file_path]
    )
    logger.info("File format validation successful.")

    enable_cot = True if enable_cot_str.lower() == "true" else False
    enable_cod = True if enable_cod_str.lower() == "true" else False

    mlclient_ws = get_workspace_mlclient()
    if not mlclient_ws:
        raise Exception("Could not create MLClient for current workspace")

    if teacher_model_endpoint_name:
        endpoint_details = get_endpoint_details(
            mlclient_ws, teacher_model_endpoint_name
        )
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
            f"Invalid teacher_model_top_p. Value should be 0<=val<=1, but it is {teacher_model_top_p}"
        )
    if teacher_model_temperature < 0 or teacher_model_temperature > 1:
        raise Exception(
            f"Invalid teacher_model_temperature. Value should be 0<=val<=1, but it is {teacher_model_temperature}"
        )
    inference_params = {
        MAX_NEW_TOKENS: (
            DEFAULT_SUMMARY_MAX_NEW_TOKENS
            if data_generation_task_type == "SUMMARIZATION"
            and teacher_model_max_new_tokens == DEFAULT_MAX_NEW_TOKENS
            else teacher_model_max_new_tokens
        ),
        TEMPERATURE: teacher_model_temperature,
        TOP_P: teacher_model_top_p,
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

    try:
        guid = uuid.uuid4()
        short_guid = str(guid)[:8]
        connection_name = f"distillation-ws-connection-{short_guid}"
        mlclient_ws.connections.create_or_update(
            ServerlessConnection(
                name=connection_name,
                endpoint=teacher_model_endpoint_url,
                api_key=teacher_model_endpoint_key,
            )
        )
        logger.info(f"Connection created with name: {connection_name}")
        config = {}
        config["scoring_url"] = teacher_model_endpoint_url
        config["connection_name"] = connection_name
        with open(batch_config_connection, "w") as f:
            json.dump(config, f)
    except Exception as e:
        logger.error(
            f"Failed to create connection for teacher model batch score invocation : {e}"
        )
        raise Exception(
            "Failed to create workspace connection for teacher model batch score invocation "
        )

    logger.info("Running data preprocessing")
    preprocess_data(
        inference_params=inference_params,
        enable_cot=enable_cot,
        enable_cod=enable_cod,
        max_len_summary=max_len_summary,
        generated_train_payload_path=generated_train_payload_path,
        generated_validation_payload_path=generated_validation_payload_path,
        train_file_path=train_file_path,
        data_generation_task_type=data_generation_task_type,
        validation_file_path=validation_file_path,
        hash_train_data=hash_train_data,
        hash_validation_data=hash_validation_data,
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
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    data_import(args)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for FTaaS data import component."""

import json
import logging

import argparse
from argparse import Namespace
from pathlib import Path
from typing import List, Dict, Any

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
from common.io import read_jsonl_files
from common.constants import (
    COMPONENT_NAME,
    DEFAULT_SUCCESS_RATIO,
    DataGenerationTaskType,
    HashField,
    TelemetryConstants,
    SystemPrompt,
    PayloadField,
    STATUS_SUCCESS,
    FINISH_REASON_STOP,
)
# from common.student_models import StudentModels

from common.utils import (
    get_hash_value,
    get_workspace_mlclient,
)

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
        "--batch_score_train_result",
        type=str,
        help="Path to the directory containing jsonl file(s) that have the result for each payload.",
    )

    parser.add_argument(
        "--batch_score_validation_result",
        default=None,
        type=str,
        help="Path to the directory containing jsonl file(s) that have the result for each payload.",
    )

    parser.add_argument(
        "--hash_train_data",
        type=str,
        required=True,
        help="Path tho the jsonl file containing the hash for each payload.",
    )

    parser.add_argument(
        "--hash_validation_data",
        type=str,
        default=None,
        help="Path tho the jsonl file containing the hash for each payload.",
    )

    parser.add_argument(
        "--generated_batch_train_file_path",
        type=Path,
        default=None,
        help="file to save the generated training data",
    )

    parser.add_argument(
        "--generated_batch_validation_file_path",
        type=Path,
        default=None,
        help="file to save the generated validation data",
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
        ),
    )

    parser.add_argument(
        "--connection_config_file",
        type=str,
        required=False,
        default=None,
        help="A config file path that contains deployment configurations.",
    )

    # parser.add_argument(
    #     "--model_asset_id",
    #     type=str,
    #     required=True,
    #     help="The student model asset id"
    # )

    return parser


def delete_connection(config_file: str):
    """Delete the connection configuration file.

    Args:
        config_file (str): The path to the connection configuration file.
    """
    if config_file:
        try:
            config_from_file = {}
            with open(config_file) as file:
                config_from_file = json.load(file)
            batch_connection_name = config_from_file.get("connection_name", None)
            if batch_connection_name:
                mlclient_ws = get_workspace_mlclient()
                if not mlclient_ws:
                    raise Exception("Could not create MLClient for current workspace")
                mlclient_ws.connections.delete(batch_connection_name)
        except Exception as e:
            msg = f"Error deleting connection: {e}"
            logger.error(msg)
            raise Exception(msg)


def postprocess_data(
    batch_score_res_path: str,
    input_file_path: str,
    enable_cot: bool,
    enable_cod: bool,
    data_generation_task_type: str,
    min_endpoint_success_ratio: float,
    output_file_path: str,
    hash_data: str
    # student_model: str
):
    """Generate and save synthentic data under output_dataset.

    Args:
        batch_score_res_path (str): Path containing jsonl file(s) that have the result for each payload.
        input_file_path (str): Input JSONL file path.
        enable_cot (bool): Enable Chain of Thought
        enable_cod (bool): Enable Chain of Density
        data_generation_task_type (str): Data generation task type
        min_endpoint_success_ratio (float): Minimum success ratio below which run will be considered a failure
        output_file_path (str): Output JSONL file path.
        hash_data (str): Path to the jsonl file containing the hash for each payload.
        student_model (str): The student model to finetune
    """
    error_count = 0
    output_data = []
    if input_file_path is None:
        logger.info(
            f"No input file path provided. Skipping data postprocessing for {input_file_path}."
        )
        return
    hash_data_list: List[Dict[str, Any]] = read_jsonl_files(hash_data)
    # Recreate hash_data_list respecting the order of batch_score_res_list
    hash_data_dict = {
        hash_data[HashField.HASH]: hash_data for hash_data in hash_data_list
    }
    total_rows = len(hash_data_list)

    batch_score_res_list: List[Dict[str, Any]] = read_jsonl_files(batch_score_res_path)
    try:
        for idx, batch_score_dict in enumerate(batch_score_res_list):
            status = batch_score_dict.get("status", "")
            if status == STATUS_SUCCESS:
                request_dict = batch_score_dict.get("request", {})
                if request_dict:
                    synthetic_responses = []
                    messages = request_dict.pop("messages", [])
                    for message in messages:
                        role = message["role"]
                        if role == PayloadField.USER:
                            hash_val = get_hash_value(message)
                            system_message = hash_data_dict[hash_val].get(
                                PayloadField.SYSTEM, {}
                            )
                            synthetic_responses.append(system_message)
                            synthetic_responses.append(message)
                            break
                    response_data = batch_score_dict.get("response", {})
                    finish_reason = response_data["choices"][0]["finish_reason"]
                    if finish_reason == FINISH_REASON_STOP:
                        prediction_result = response_data["choices"][0]["message"][
                            "content"
                        ].strip()
                        # For CoT prompts, need to remove the reasoning and only use the answer
                        if (
                            enable_cot
                            and data_generation_task_type
                            != DataGenerationTaskType.CONVERSATION
                        ):
                            key = SystemPrompt.get_response_key(
                                data_generation_task_type
                            )
                            prediction_result = json.loads(prediction_result)[key]

                        if (
                            enable_cod
                            and data_generation_task_type
                            == DataGenerationTaskType.SUMMARIZATION
                        ):
                            result = json.loads(prediction_result)
                            prediction_result = result[-1]["Denser_Summary"]

                        synthetic_responses.append(
                            {"role": "assistant", "content": str(prediction_result)}
                        )
                        output_data.append({"messages": synthetic_responses})
                    else:
                        error_count += 1
            else:
                error_count += 1
    except Exception as e:
        logger.error(f"Error in postprocessing {idx} data: {e}")
        raise e
    success_ratio = float(total_rows - error_count) / total_rows
    print(success_ratio)
    if success_ratio < min_endpoint_success_ratio:
        msg = f"Success ratio for dataset {input_file_path}: {success_ratio} < {min_endpoint_success_ratio}."
        raise Exception(msg)

    # Reformat data based on student model limitations
    # output_data = StudentModels.reformat(
    #   student_model=student_model,
    #   task_type=data_generation_task_type,
    #   data=output_data
    # )
    with open(output_file_path, "w") as f:
        for record in output_data:
            f.write(json.dumps(record) + "\n")


def data_import(args: Namespace):
    """Copy the user data to output dir."""
    train_file_path = args.train_file_path
    validation_file_path = args.validation_file_path
    batch_score_train_path = args.batch_score_train_result
    batch_score_validation_path = args.batch_score_validation_result
    generated_batch_train_file_path = args.generated_batch_train_file_path
    generated_batch_validation_file_path = args.generated_batch_validation_file_path
    enable_cot_str = args.enable_chain_of_thought
    enable_cod_str = args.enable_chain_of_density
    min_endpoint_success_ratio = args.min_endpoint_success_ratio
    data_generation_task_type = args.data_generation_task_type
    hash_train_data = args.hash_train_data
    hash_validation_data = args.hash_validation_data
    connection_config_file = args.connection_config_file
    # model_asset_id = args.model_asset_id

    enable_cot = True if enable_cot_str.lower() == "true" else False
    enable_cod = True if enable_cod_str.lower() == "true" else False

    with log_activity(
        logger=logger, activity_name=TelemetryConstants.POST_PROCESS_TRAINING_DATA
    ):
        logger.info(
            "Deleting batch configuration connection used for teacher model invocation."
        )
        delete_connection(connection_config_file)
        logger.info(
            "Running data postprocessing for train file path: %s", train_file_path
        )
        postprocess_data(
            batch_score_res_path=batch_score_train_path,
            input_file_path=train_file_path,
            enable_cot=enable_cot,
            enable_cod=enable_cod,
            data_generation_task_type=data_generation_task_type,
            min_endpoint_success_ratio=min_endpoint_success_ratio,
            output_file_path=generated_batch_train_file_path,
            hash_data=hash_train_data
            # student_model=StudentModels.parse_model_asset_id(model_asset_id)
        )
    if validation_file_path:
        with log_activity(
            logger=logger,
            activity_name=TelemetryConstants.POST_PROCESS_VALIDATION_DATA,
        ):
            logger.info(
                "Running data postprocessing for validation file path: %s",
                validation_file_path,
            )
            postprocess_data(
                batch_score_res_path=batch_score_validation_path,
                input_file_path=validation_file_path,
                enable_cot=enable_cot,
                enable_cod=enable_cod,
                data_generation_task_type=data_generation_task_type,
                min_endpoint_success_ratio=min_endpoint_success_ratio,
                output_file_path=generated_batch_validation_file_path,
                hash_data=hash_validation_data
                # student_model=StudentModels.parse_model_asset_id(model_asset_id)
            )
    else:
        Path(generated_batch_validation_file_path.parent).mkdir(
            exist_ok=True, parents=True
        )
        # create an empty file if validation file is not provided
        open(generated_batch_validation_file_path, "w").close()


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

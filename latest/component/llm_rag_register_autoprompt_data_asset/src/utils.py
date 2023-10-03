# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Helper File for the autoprompt utility methods."""
import pandas as pd
import copy
import logging
from run_utils import current_run, TestRun
from typing import Callable, Any
from azureml.rag.utils.connections import get_connection_by_id_v2, workspace_connection_to_credential
from constants import TaskTypes
from logging_utilities import get_logger, log_info, log_traceback, log_warning
from error_definitions import (
    BadInputData,
    InvalidAnswersKey,
    InvalidContextKey,
    InvalidQuestionsKey
)
from exceptions import DataLoaderException, ArgumentValidationException

import openai
import openai.error
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import json
import os
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml._common._error_definition.error_definition import ErrorDefinition
from azureml.exceptions import AzureMLException

logger = get_logger("utils")


def get_custom_dimensions():
    """Get Custom Dimensions for Activity logging."""
    run = TestRun()
    custom_dimensions = {
        "app_name": "autoprompt",
        "run_id": run.run.id,
        "parent_run_id": run.root_run.id,
        "experiment_id": run.experiment.name,
        "workspace": run.workspace.name,
        "subscription": run.subscription,
        "target": run.compute,
        "region": run.region
    }
    return custom_dimensions


CUSTOM_DIMENSIONS = get_custom_dimensions()


def make_and_log_exception(error_cls: ErrorDefinition,
                           exception_cls: AzureMLException,
                           logger,
                           inner_exception: Exception,
                           activity_logger=None,
                           message_params={}):
    """Create and log custom exception to app insights.

    Args:
        error_cls (ErrorDefinition): _description_
        exception_cls (AzureMLException): _description_
        logger (_type_): _description_
        inner_exception (Exception): _description_
        activity_logger (_type_, optional): _description_. Defaults to None.
        message_params (dict, optional): _description_. Defaults to {}.

    Raises:
        exception_obj: _description_
    """
    aml_error = AzureMLError.create(error_cls, **message_params)
    exception_obj = exception_cls._with_error(
        azureml_error=aml_error,
        inner_exception=inner_exception
    )
    log_traceback(exception_obj, logger, CUSTOM_DIMENSIONS)
    if activity_logger:
        activity_logger.activity_info['error'] = f"{inner_exception.__class__.__name__}: {exception_obj.message}"
        activity_logger.exception(exception_obj.message)
    raise exception_obj


def openai_init(llm_config, **openai_params):
    """Initialize OpenAI Params."""
    log_info(logger, f"Using llm_config: {json.dumps(llm_config, indent=2)}", CUSTOM_DIMENSIONS)
    openai_api_type = openai_params.get("openai_api_type", "azure")
    openai_api_version = openai_params.get("openai_api_version", "2023-03-15-preview")

    connection_id = os.environ.get('AZUREML_WORKSPACE_CONNECTION_ID_AOAI', None)
    fetch_from_connection = False
    if connection_id is not None:
        connection = get_connection_by_id_v2(connection_id)
        credential = workspace_connection_to_credential(connection)
        if hasattr(credential, 'key'):
            llm_config["key"] = credential.key
            llm_config["base"] = connection.target
            connection_metadata = connection.metadata
            openai_api_type = connection_metadata.get(
                'apiType',
                connection_metadata.get('ApiType', "azure"))
            openai_api_version = connection_metadata.get(
                'apiVersion',
                connection_metadata.get('ApiVersion', "2023-03-15-preview"))
            log_info(logger, "Using workspace connection key for OpenAI", CUSTOM_DIMENSIONS)
            fetch_from_connection = True
    if not fetch_from_connection:
        if llm_config.get("type") == "azure_open_ai":
            ws = current_run.experiment.workspace
            keyvault = ws.get_default_keyvault()
            secrets = keyvault.get_secrets(secrets=[
                "BAKER-OPENAI-API-BASE",
                "BAKER-OPENAI-API-KEY",
                "OPENAI-API-KEY",
                "OPENAI-API-BASE"])
            log_info(logger, "Run context and secrets retrieved", CUSTOM_DIMENSIONS)

            # hacky way to override OPENAI-API-KEY if Baker key existed
            if secrets["BAKER-OPENAI-API-BASE"] is not None:
                secrets["OPENAI-API-BASE"] = secrets["BAKER-OPENAI-API-BASE"]
            if secrets["BAKER-OPENAI-API-KEY"] is not None:
                secrets["OPENAI-API-KEY"] = secrets["BAKER-OPENAI-API-KEY"]
            llm_config["key"] = secrets["OPENAI-API-KEY"]
            llm_config["base"] = secrets["OPENAI-API-BASE"]
        else:
            raise NotImplementedError(f"LLM type '{llm_config['type']}' not supported!")

    openai.api_version = openai_api_version
    openai.api_type = openai_api_type
    openai.api_base = llm_config["base"]
    openai.api_key = llm_config["key"]

    openai_final_params = {
        "api_version": openai_api_version,
        "api_type": openai_api_type,
        "api_base": llm_config["base"],
        "api_key": llm_config["key"],
        "deployment_id": llm_config['deployment_name']
    }
    return openai_final_params


def _validate_keys(df, questions_key, answers_key, context_key=None):
    if any([q_key not in df.columns for q_key in questions_key]):
        make_and_log_exception(
            error_cls=InvalidQuestionsKey,
            exception_cls=ArgumentValidationException,
            logger=logger,
            inner_exception=ValueError(f"[{questions_key}] not in input data."),
            activity_logger=None,
            message_params={"questions_key": str(questions_key)}
        )

    if answers_key not in df.columns:
        make_and_log_exception(
            error_cls=InvalidAnswersKey,
            exception_cls=ArgumentValidationException,
            logger=logger,
            inner_exception=ValueError(f"[{answers_key}] not in input data."),
            activity_logger=None,
            message_params={"answers_key": str(answers_key)}
        )

    if context_key is not None:
        if context_key not in df.columns:
            make_and_log_exception(
                error_cls=InvalidContextKey,
                exception_cls=ArgumentValidationException,
                logger=logger,
                inner_exception=ValueError(f"[{context_key}] not in input data.\
                                           For mcq task context_key represents your choices key in dataset."),
                activity_logger=None,
                message_params={"answers_key": str(context_key)}
            )


def read_data(task_type, dev_data_path, test_data_path, text_keys, label_key, context_key):
    """Read input data."""
    num_choices = 0
    log_info(logger, "Reading Data", CUSTOM_DIMENSIONS)
    if task_type == TaskTypes.mcq:
        df_dev, y_true_dev, num_choices_dev = read_mcq_data(dev_data_path, text_keys, label_key, context_key)
        df_test, y_true_test, num_choices_test = read_mcq_data(test_data_path, text_keys, label_key, context_key)
        if num_choices_test != num_choices_dev:
            raise ValueError("Number of choices mismatch between test dataset and dev dataset.")
        num_choices = num_choices_dev
    elif task_type == TaskTypes.abstractive:
        df_dev, y_true_dev = read_squad_data(dev_data_path, text_keys, label_key, context_key)
        df_test, y_true_test = read_squad_data(test_data_path, text_keys, label_key, context_key)
    else:
        df_dev, y_true_dev = read_arithmetic_data(dev_data_path, text_keys, label_key)
        df_test, y_true_test = read_arithmetic_data(test_data_path, text_keys, label_key)
    return df_dev, y_true_dev, df_test, y_true_test, num_choices


def read_arithmetic_data(test_data_path, text_keys, label_key):
    """Read Data."""
    try:
        data_df = pd.read_json(test_data_path, lines=True, dtype=False)
    except Exception as e:
        make_and_log_exception(
            error_cls=BadInputData,
            exception_cls=DataLoaderException,
            logger=logger,
            activity_logger=None,
            inner_exception=e,
            message_params={"error": repr(e)}
        )
    _validate_keys(data_df, text_keys, label_key)
    X, y = data_df[text_keys], data_df[label_key]
    return X, y


def convert_choices(x):
    """Convert Choices."""
    prompt = "\nAnswer Choices:"
    for label, text in zip(x["label"], x["text"]):
        prompt += " ("+label+") "+text
    return prompt


def read_mcq_data(test_data_path, text_keys, label_key, choices_key):
    """Read Multi-Choice Question Data."""
    try:
        data_df = pd.read_json(test_data_path, lines=True, dtype=False)
    except Exception as e:
        make_and_log_exception(
            error_cls=BadInputData,
            exception_cls=DataLoaderException,
            logger=logger,
            activity_logger=None,
            inner_exception=e,
            message_params={"error": repr(e)}
        )

    _validate_keys(data_df, text_keys, label_key, choices_key)
    data, y = data_df[text_keys+[choices_key]], data_df[label_key]
    X = pd.DataFrame(
        data.apply(
            lambda x: x[text_keys[0]] + convert_choices(x[choices_key]),
            axis=1),
        columns=text_keys)
    num_choices = max(data_df[choices_key].apply(lambda x: len(x["label"])))
    return X, y, num_choices


def read_squad_data(test_data_path, text_keys, label_key, context_key):
    """Read Squad Data."""
    try:
        data_df = pd.read_json(test_data_path, lines=True, dtype=False)
    except Exception as e:
        make_and_log_exception(
            error_cls=BadInputData,
            exception_cls=DataLoaderException,
            logger=logger,
            activity_logger=None,
            inner_exception=e,
            message_params={"error": repr(e)}
        )

    _validate_keys(data_df, text_keys, label_key, context_key)
    data, y = data_df[text_keys+[context_key]], data_df[label_key]
    X = pd.DataFrame(
        data.apply(
            lambda x: "Context: "+x[context_key]+"\nQuestion: "+x[text_keys[0]],
            axis=1),
        columns=text_keys)
    return X, y


def _create_retry_decorator() -> Callable[[Any], Any]:
    max_retries = 6
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(**kwargs: Any) -> Any:
    """Use tenacity to retry the completion call.

    Copied from: https://github.com/hwchase17/langchain/blob/42df78d3964170bab39d445aa2827dea10a312a7 \
        /langchain/llms/openai.py#L98
    """
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        model_name = kwargs['model']
        if model_name.startswith("gpt-3.5-turbo") \
            or model_name.startswith("gpt-35-turbo") \
                or model_name.startswith("gpt-4"):
            kwargs['messages'] = [{'role': 'system', 'content': kwargs['prompt'][0]}]
            kwargs_copy = copy.deepcopy(kwargs)
            del kwargs_copy['prompt']
            if 'best_of' in kwargs_copy:
                del kwargs_copy['best_of']
            return openai.ChatCompletion.create(**kwargs_copy)
        else:
            return openai.Completion.create(**kwargs)

    return _completion_with_retry(**kwargs)


def get_predictions(data, max_tokens=500, batch_size=20, temperature=0.0, **kwargs):
    """Get Predictions."""
    log_info(
        logger,
        f"Max Tokens: {max_tokens}, Batch Size: {batch_size}, Temperature:{temperature}",
        CUSTOM_DIMENSIONS)
    y_pred = [""]*len(data)
    model_name = kwargs['llm_config']['model_name']
    if model_name.startswith("gpt-3.5-turbo") \
        or model_name.startswith("gpt-35-turbo") \
            or model_name.startswith("gpt-4"):
        chat = True
    else:
        chat = False

    for i in range(0, len(data), batch_size):
        log_info(logger, "Procesing batch:"+str(i)+"-"+str(i+batch_size), CUSTOM_DIMENSIONS)
        data_batch = list(data[i:i + batch_size])
        try:
            out = completion_with_retry(
                model=model_name,
                deployment_id=kwargs['llm_config']['deployment_name'],
                prompt=data_batch,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except openai.error.InvalidRequestError:
            log_warning(
                logger,
                "Content filter warning encountered. Going via single prompt and skipping filtered results",
                CUSTOM_DIMENSIONS)
            out = {"choices": []}
            for j in range(i, i+batch_size):
                try:
                    cur_out = completion_with_retry(
                        model=model_name,
                        deployment_id=kwargs['llm_config']['deployment_name'],
                        prompt=data[j],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if chat:
                        out["choices"].append({
                            "message": {
                                "content": cur_out["choices"][0]["text"],
                                "role": "system"
                            },
                            "index": j-i
                        })
                    else:
                        out["choices"].append({"text": cur_out["choices"][0]["text"], "index": j-i})
                except openai.error.InvalidRequestError:
                    if chat:
                        out["choices"].append({
                            "message": {
                                "content": "could_not_classify",
                                "role": "system"
                            },
                            "index": j-i
                        })
                    else:
                        out["choices"].append({"text": "could_not_classify", "index": j-i})
            break
        # pprint.pprint(out)
        # break

        # Collect predictions from response
        for idx, row in enumerate(out["choices"]):
            # print(row["message"])
            text = row["message"]["content"].strip().lower() if chat else row["text"].strip().lower()
            if text.endswith("."):
                text = text[:-1]
            y_pred[i + row["index"]] = text

    return y_pred

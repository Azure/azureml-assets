# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Union

from urllib.parse import urlparse

import mlflow
import mltable

import pandas as pd

from spacy.cli import download

from responsibleai import __version__ as responsibleai_version
from responsibleai.feature_metadata import FeatureMetadata

from responsibleai_text import (
    RAITextInsights,
    __version__ as responsibleai_text_version,
)
from raiutils.data_processing import serialize_json_safe

from azure.core.configuration import Configuration
from azure.core.pipeline import Pipeline
from azure.core.pipeline.policies import (
    CustomHookPolicy,
    HeadersPolicy,
    HttpLoggingPolicy,
    NetworkTraceLoggingPolicy,
    ProxyPolicy,
    RedirectPolicy,
    RetryPolicy,
    UserAgentPolicy,
)
from azure.core.pipeline.transport import RequestsTransport
from azure.core.rest import HttpRequest
from azureml.rai.utils import ModelSerializer
from azureml.rai.utils.telemetry import LoggerFactory, track

# Local
from constants import TaskType

print(download("en_core_web_sm"))

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


TEXT_DATA_TYPE = "text"
COMPONENT_NAME = "azureml.rai.text"
DEFAULT_MODULE_NAME = "rai_text_insights"
DEFAULT_MODULE_VERSION = "0.0.0"

# Taken from azure-ai-evaluation _eval_run.py file
_MAX_RETRIES = 5
_BACKOFF_FACTOR = 2
_TIMEOUT = 5
_POLLING_INTERVAL = 30

_ai_logger = None
_module_name = DEFAULT_MODULE_NAME
_module_version = DEFAULT_MODULE_VERSION


def _get_logger():
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = LoggerFactory.get_logger(
            __file__, _module_name, _module_version, COMPONENT_NAME)
    return _ai_logger


class ModelTypes:
    FASTAI = "fastai"
    PYFUNC = "pyfunc"
    PYTORCH = "pytorch"
    HFTRANSFORMERS = "hftransformers"


class DashboardInfo:
    MODEL_ID_KEY = "id"  # To match Model schema
    MODEL_INFO_FILENAME = "model_info.json"
    TRAIN_FILES_DIR = "train"
    TEST_FILES_DIR = "test"

    RAI_INSIGHTS_MODEL_ID_KEY = "model_id"
    RAI_INSIGHTS_RUN_ID_KEY = "rai_insights_parent_run_id"
    RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY = "constructor_args"
    RAI_INSIGHTS_PARENT_FILENAME = "rai_insights.json"


class PropertyKeyValues:
    # The property to indicate the type of Run
    RAI_INSIGHTS_TYPE_KEY = "_azureml.responsibleai.rai_insights.type"
    RAI_INSIGHTS_TYPE_CONSTRUCT = "construction"
    RAI_INSIGHTS_TYPE_CAUSAL = "causal"
    RAI_INSIGHTS_TYPE_COUNTERFACTUAL = "counterfactual"
    RAI_INSIGHTS_TYPE_EXPLANATION = "explanation"
    RAI_INSIGHTS_TYPE_ERROR_ANALYSIS = "error_analysis"
    RAI_INSIGHTS_TYPE_GATHER = "gather"

    # Property to point at the model under examination
    RAI_INSIGHTS_MODEL_ID_KEY = "_azureml.responsibleai.rai_insights.model_id"

    # Property for tool runs to point at their constructor run
    RAI_INSIGHTS_CONSTRUCTOR_RUN_ID_KEY = (
        "_azureml.responsibleai.rai_insights.constructor_run"
    )

    # Property to record responsibleai version
    RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY = (
        "_azureml.responsibleai.rai_insights.responsibleai_version"
    )

    # Property format to indicate presence of a tool
    RAI_INSIGHTS_TOOL_KEY_FORMAT = \
        "_azureml.responsibleai.rai_insights.has_{0}"

    RAI_INSIGHTS_TEXT_VERSION_KEY = "responsibleai_text_version"

    RAI_INSIGHTS_DATA_TYPE_KEY = (
        "_azureml.responsibleai.rai_insights.data_type"
    )


class RAIToolType:
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    ERROR_ANALYSIS = "error_analysis"
    EXPLANATION = "explanation"


def boolean_parser(target: str) -> bool:
    true_values = ["True", "true"]
    false_values = ["False", "false"]
    if target in true_values:
        return True
    if target in false_values:
        return False
    raise ValueError("Failed to parse to boolean: {target}")


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--task_type", type=str, required=True,
        choices=[TaskType.TEXT_CLASSIFICATION,
                 TaskType.MULTILABEL_TEXT_CLASSIFICATION,
                 TaskType.QUESTION_ANSWERING]
    )

    parser.add_argument(
        "--model_input", type=str, help="model local path on remote",
        required=True
    )

    parser.add_argument(
        "--model_info", type=str, help="name:version", required=True
    )

    parser.add_argument("--test_dataset", type=str, required=True)

    parser.add_argument("--target_column_name", type=str, required=True)
    parser.add_argument("--text_column_name", type=str, required=False)

    parser.add_argument("--maximum_rows_for_test_dataset", type=int,
                        default=5000)
    parser.add_argument(
        "--categorical_column_names", type=str, help="Optional[List[str]]"
    )

    parser.add_argument("--classes", type=str, help="Optional[List[str]]")

    parser.add_argument("--categorical_metadata_features", type=str,
                        help="Optional[List[str]]")

    parser.add_argument("--dropped_metadata_features", type=str,
                        help="Optional[List[str]]")

    # Explanations
    parser.add_argument("--enable_explanation", type=boolean_parser)

    # Error analysis
    parser.add_argument("--enable_error_analysis", type=boolean_parser)

    parser.add_argument("--use_model_dependency", type=boolean_parser,
                        help="Use model dependency")
    parser.add_argument("--use_conda", type=boolean_parser,
                        help="Use conda instead of pip")

    # Component info
    parser.add_argument("--component_name", type=str, required=True)
    parser.add_argument("--component_version", type=str, required=True)

    # Outputs
    parser.add_argument("--dashboard", type=str, required=True)
    parser.add_argument("--ux_json", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def get_from_args(args, arg_name: str, custom_parser, allow_none: bool) -> Any:
    _logger.info("Looking for command line argument '{0}'".format(arg_name))
    result = None

    extracted = getattr(args, arg_name)
    if extracted is None and not allow_none:
        raise ValueError("Required argument {0} missing".format(arg_name))

    if custom_parser:
        if extracted is not None:
            result = custom_parser(extracted)
    else:
        result = extracted

    _logger.info("{0}: {1}".format(arg_name, result))

    return result


def json_empty_is_none_parser(target: str) -> Union[Dict, List]:
    parsed = json.loads(target)
    if len(parsed) == 0:
        return None
    else:
        return parsed


def load_mltable(mltable_path: str) -> pd.DataFrame:
    _logger.info("Loading MLTable: {0}".format(mltable_path))
    df: pd.DataFrame = None
    try:
        tbl = mltable.load(mltable_path)
        df: pd.DataFrame = tbl.to_pandas_dataframe()
    except Exception as e:
        _logger.info("Failed to load MLTable")
        _logger.info(e)
    return df


def load_parquet(parquet_path: str) -> pd.DataFrame:
    _logger.info("Loading parquet file: {0}".format(parquet_path))
    df = pd.read_parquet(parquet_path)
    return df


def load_dataset(dataset_path: str) -> pd.DataFrame:
    _logger.info(f"Attempting to load: {dataset_path}")
    df = load_mltable(dataset_path)
    if df is None:
        df = load_parquet(dataset_path)
    print(df.dtypes)
    print(df.head(10))
    return df


def fetch_model_id(model_info_path: str):
    model_info_path = os.path.join(model_info_path,
                                   DashboardInfo.MODEL_INFO_FILENAME)
    with open(model_info_path, "r") as json_file:
        model_info = json.load(json_file)
    return model_info[DashboardInfo.MODEL_ID_KEY]


def get_http_client():
    config = Configuration()
    config.headers_policy = HeadersPolicy()
    config.proxy_policy = ProxyPolicy()
    config.redirect_policy = RedirectPolicy()
    config.retry_policy = RetryPolicy(
                retry_total=_MAX_RETRIES,
                retry_connect=_MAX_RETRIES,
                retry_read=_MAX_RETRIES,
                retry_status=_MAX_RETRIES,
                retry_on_status_codes=(408, 429, 500, 502, 503, 504),
                retry_backoff_factor=_BACKOFF_FACTOR)
    config.custom_hook_policy = CustomHookPolicy()
    config.logging_policy = NetworkTraceLoggingPolicy()
    config.http_logging_policy = HttpLoggingPolicy()
    config.user_agent_policy = UserAgentPolicy()
    config.polling_interval = _POLLING_INTERVAL
    return Pipeline(
        transport=RequestsTransport(),
        policies=[
            config.headers_policy,
            config.user_agent_policy,
            config.proxy_policy,
            config.redirect_policy,
            config.retry_policy,
            config.custom_hook_policy,
            config.logging_policy,
        ]
    )


def get_scope() -> str:
    """
    Return the scope information for the workspace.

    :return: The scope information for the workspace.
    :rtype: str
    """
    workspace_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    return (
        "/subscriptions/{}/resourceGroups/{}/providers"
        "/Microsoft.MachineLearningServices"
        "/workspaces/{}"
    ).format(
        subscription_id,
        resource_group,
        workspace_name,
    )


def get_run_history_uri(tracking_uri: str, run_id: str,
                        experiment_id: str) -> str:
    """
    Get the run history service URI.

    :return: The run history service URI.
    :rtype: str
    """
    url_base = urlparse(tracking_uri).netloc
    return (
        f"https://{url_base}"
        "/history/v1.0"
        f"{get_scope()}"
        f"/experimentids/{experiment_id}/runs/{run_id}"
    )


def add_properties_to_gather_run(
    dashboard_info: Dict[str, str], tool_present_dict: Dict[str, str]
):
    _logger.info("Adding properties to the gather run")

    run_properties = {
        PropertyKeyValues.RAI_INSIGHTS_TYPE_KEY:
            PropertyKeyValues.RAI_INSIGHTS_TYPE_GATHER,
        PropertyKeyValues.RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY:
            responsibleai_version,
        PropertyKeyValues.RAI_INSIGHTS_TEXT_VERSION_KEY:
            responsibleai_text_version,
        PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_DATA_TYPE_KEY:
            TEXT_DATA_TYPE
    }

    _logger.info("Appending tool present information")
    for k, v in tool_present_dict.items():
        key = PropertyKeyValues.RAI_INSIGHTS_TOOL_KEY_FORMAT.format(k)
        run_properties[key] = str(v)

    _logger.info("Making service call from MlflowClient")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        json_dict = {"runId": run_id, "properties": run_properties}
        headers = {}
        headers["User-Agent"] = _module_name + "(" + _module_version + ")"
        token = os.environ.get("MLFLOW_TRACKING_TOKEN")
        headers["Authorization"] = f"Bearer {token}"

        session = get_http_client()
        method = "PATCH"
        url = get_run_history_uri(mlflow.get_tracking_uri(), run_id,
                                  experiment_id)
        request = HttpRequest(method, url, headers=headers, json=json_dict)
        response = session.run(request, timeout=_TIMEOUT).http_response
        _logger.info("Response from run history on adding properties: ",
                     response)

    _logger.info("Properties added to gather run")


@track(_get_logger)
def main(args):
    _logger.info("Loading evaluation dataset")
    test_df = load_dataset(args.test_dataset)

    if args.model_input is None or args.model_info is None:
        raise ValueError(
            "Both model info and model input need to be supplied.")

    model_meta = mlflow.models.Model.load(
                    os.path.join(args.model_input, "MLmodel"))
    model_type = None
    if ModelTypes.HFTRANSFORMERS in model_meta.flavors:
        model_type = ModelTypes.HFTRANSFORMERS

    model_id = args.model_info
    _logger.info("Loading model: {0}".format(model_id))

    tracking_uri = mlflow.get_tracking_uri()
    model_serializer = ModelSerializer(
        model_id,
        mlflow_model=args.model_input,
        model_type=model_type,
        use_model_dependency=args.use_model_dependency,
        use_conda=args.use_conda,
        task_type=args.task_type,
        tracking_uri=tracking_uri)
    text_model = model_serializer.load("Ignored path")

    text_column = args.text_column_name

    if args.task_type == TaskType.MULTILABEL_TEXT_CLASSIFICATION:
        target_column = get_from_args(
            args=args,
            arg_name="target_column_name",
            custom_parser=json_empty_is_none_parser,
            allow_none=True
        )
    else:
        target_column = args.target_column_name

    feature_metadata = FeatureMetadata()
    feature_metadata.categorical_features = get_from_args(
        args=args,
        arg_name="categorical_metadata_features",
        custom_parser=json_empty_is_none_parser,
        allow_none=True
    )
    feature_metadata.dropped_features = get_from_args(
        args=args,
        arg_name="dropped_metadata_features",
        custom_parser=json_empty_is_none_parser,
        allow_none=True
    )

    _logger.info("Creating RAI Text Insights")
    rai_ti = RAITextInsights(
        model=text_model,
        test=test_df,
        target_column=target_column,
        task_type=args.task_type,
        classes=get_from_args(
            args, "classes", custom_parser=json_empty_is_none_parser,
            allow_none=True
        ),
        maximum_rows_for_test=args.maximum_rows_for_test_dataset,
        serializer=model_serializer,
        feature_metadata=feature_metadata,
        text_column=text_column
    )

    included_tools: Dict[str, bool] = {
        RAIToolType.CAUSAL: False,
        RAIToolType.COUNTERFACTUAL: False,
        RAIToolType.ERROR_ANALYSIS: False,
        RAIToolType.EXPLANATION: False,
    }

    if args.enable_explanation:
        _logger.info("Adding explanation")
        rai_ti.explainer.add()
        included_tools[RAIToolType.EXPLANATION] = True

    if args.enable_error_analysis:
        _logger.info("Adding error analysis")
        rai_ti.error_analysis.add()
        included_tools[RAIToolType.ERROR_ANALYSIS] = True

    _logger.info("Starting computation")
    rai_ti.compute()
    _logger.info("Computation complete")

    rai_ti.save(args.dashboard)
    _logger.info("Saved dashboard to output")

    rai_data = rai_ti.get_data()
    rai_dict = serialize_json_safe(rai_data)
    json_filename = "dashboard.json"
    output_path = Path(args.ux_json) / json_filename
    with open(output_path, "w") as json_file:
        json.dump(rai_dict, json_file)
    _logger.info("Dashboard JSON written")

    dashboard_info = dict()
    dashboard_info[DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY] = model_id

    add_properties_to_gather_run(dashboard_info, included_tools)
    _logger.info("Processing completed")


# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()
    print("Arguments parsed successfully")
    print(args)
    _module_name = args.component_name
    _module_version = args.component_version
    _get_logger()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")

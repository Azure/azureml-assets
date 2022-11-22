# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Union

import mltable

from azureml.core import Run

import pandas as pd

from spacy.cli import download

from responsibleai import __version__ as responsibleai_version

from responsibleai_text import (
    RAITextInsights,
    __version__ as responsibleai_text_version,
)
from raiutils.data_processing import serialize_json_safe

from azureml_model_serializer import ModelSerializer

print(download("en_core_web_sm"))

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


TEXT_DATA_TYPE = "text"


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
        "--task_type", type=str, required=True, choices=["text_classification"]
    )

    parser.add_argument(
        "--model_input", type=str, help="model local path on remote",
        required=True
    )

    parser.add_argument(
        "--model_info", type=str, help="name:version", required=True
    )

    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)

    parser.add_argument("--target_column_name", type=str, required=True)

    parser.add_argument("--maximum_rows_for_test_dataset", type=int,
                        default=5000)
    parser.add_argument(
        "--categorical_column_names", type=str, help="Optional[List[str]]"
    )

    parser.add_argument("--classes", type=str, help="Optional[List[str]]")

    # Explanations
    parser.add_argument("--enable_explanation", type=boolean_parser)

    # Error analysis
    parser.add_argument("--enable_error_analysis", type=boolean_parser)

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


def add_properties_to_gather_run(
    dashboard_info: Dict[str, str], tool_present_dict: Dict[str, str]
):
    _logger.info("Adding properties to the gather run")
    gather_run = Run.get_context()

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

    _logger.info("Making service call")
    gather_run.add_properties(run_properties)
    _logger.info("Properties added to gather run")


def main(args):
    _logger.info("Dealing with initialization dataset")
    train_df = load_dataset(args.train_dataset)

    _logger.info("Dealing with evaluation dataset")
    test_df = load_dataset(args.test_dataset)

    if args.model_input is None or args.model_info is None:
        raise ValueError(
            "Both model info and model input need to be supplied.")

    model_id = args.model_info
    _logger.info("Loading model: {0}".format(model_id))
    my_run = Run.get_context()
    workspace = my_run.experiment.workspace
    model_serializer = ModelSerializer(model_id, workspace=workspace)
    text_model = model_serializer.load("Ignored path")

    _logger.info("Creating RAI Text Insights")
    rai_ti = RAITextInsights(
        model=text_model,
        train=train_df,
        test=test_df,
        target_column=args.target_column_name,
        task_type=args.task_type,
        classes=get_from_args(
            args, "classes", custom_parser=json_empty_is_none_parser,
            allow_none=True
        ),
        maximum_rows_for_test=args.maximum_rows_for_test_dataset,
        serializer=model_serializer,
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

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")

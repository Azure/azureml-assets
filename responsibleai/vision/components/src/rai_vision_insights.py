# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mltable
import pandas as pd

from azureml.core import Run, Workspace
from azureml.data.abstract_dataset import AbstractDataset
from azureml.exceptions import UserErrorException
from raiutils.data_processing import serialize_json_safe

from spacy.cli import download

from azureml_model_serializer import ModelSerializer
from responsibleai import __version__ as responsibleai_version
from responsibleai_vision import RAIVisionInsights
from responsibleai_vision import __version__ as responsibleai_vision_version
from responsibleai_vision.common.constants import ExplainabilityLiterals

# Local
from aml_dataset_helper import AmlDatasetHelper
from constants import TaskType

print(download("en_core_web_sm"))

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


IMAGE_DATA_TYPE = "image"


class MLTableLiterals:
    MLTABLE = "MLTable"
    MLTABLE_RESOLVEDURI = "ResolvedUri"


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

    RAI_INSIGHTS_VISION_VERSION_KEY = "responsibleai_vision_version"

    RAI_INSIGHTS_DATA_TYPE_KEY = (
        "_azureml.responsibleai.rai_insights.data_type"
    )


class RAIToolType:
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    ERROR_ANALYSIS = "error_analysis"
    EXPLANATION = "explanation"


class ModelType:
    FASTAI = "fastai"
    PYFUNC = "pyfunc"


class DatasetStoreType:
    PRIVATE = "private"
    PUBLIC = "public"


def boolean_parser(target: str) -> bool:
    true_values = ["True", "true"]
    false_values = ["False", "false"]
    if target in true_values:
        return True
    if target in false_values:
        return False
    raise ValueError(f"Failed to parse to boolean: {target}")


def _make_arg(arg_name: str) -> str:
    return "--{}".format(arg_name)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--task_type", type=str, required=True,
        choices=[TaskType.IMAGE_CLASSIFICATION,
                 TaskType.MULTILABEL_IMAGE_CLASSIFICATION]
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
    parser.add_argument(
        "--dataset_type", type=str, required=False,
        default=DatasetStoreType.PUBLIC,
        choices=[DatasetStoreType.PRIVATE, DatasetStoreType.PUBLIC],
        help="Whether it is dataset from private datastore"
        "or from public url"
    )
    parser.add_argument("--target_column_name", type=str, required=True)
    parser.add_argument(
        "--maximum_rows_for_test_dataset", type=int,
        default=5000
    )
    parser.add_argument("--classes", type=str, help="Optional[List[str]]")

    # Explanations
    parser.add_argument("--precompute_explanation", type=boolean_parser)

    parser.add_argument("--use_model_dependency", type=boolean_parser,
                        help="Use model dependency")

    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=[ModelType.PYFUNC, ModelType.FASTAI]
    )
    # XAI args
    parser.add_argument(
        _make_arg(ExplainabilityLiterals.XAI_ALGORITHM),
        type=str,
        required=False,
        choices=[
            ExplainabilityLiterals.SHAP_METHOD_NAME,
            ExplainabilityLiterals.GUIDEDBACKPROP_METHOD_NAME,
            ExplainabilityLiterals.GUIDEDGRADCAM_METHOD_NAME,
            ExplainabilityLiterals.INTEGRATEDGRADIENTS_METHOD_NAME,
            ExplainabilityLiterals.XRAI_METHOD_NAME,
        ],
    )

    parser.add_argument(
        _make_arg(ExplainabilityLiterals.N_STEPS),
        type=int, required=False,
    )

    parser.add_argument(
        _make_arg(ExplainabilityLiterals.APPROXIMATION_METHOD),
        type=str,
        required=False,
        choices=["gausslegendre", "riemann_middle"]
    )

    parser.add_argument(
        "--xrai_fast",
        type=bool, required=False,
    )

    parser.add_argument(
        _make_arg(
            ExplainabilityLiterals.CONFIDENCE_SCORE_THRESHOLD_MULTILABEL
            ),
        type=float, required=False,
    )

    # Outputs
    parser.add_argument("--dashboard", type=str, required=True)
    parser.add_argument("--ux_json", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def get_from_args(args, arg_name: str, custom_parser, allow_none: bool) -> Any:
    _logger.info(f"Looking for command line argument '{arg_name}'")
    result = None

    extracted = getattr(args, arg_name)
    if extracted is None and not allow_none:
        raise ValueError(f"Required argument {arg_name} missing")

    if custom_parser:
        if extracted is not None:
            result = custom_parser(extracted)
    else:
        result = extracted

    _logger.info(f"{arg_name}: {result}")

    return result


def json_empty_is_none_parser(target: str) -> Union[Dict, List]:
    parsed = json.loads(target)
    if len(parsed) == 0:
        return None
    else:
        return parsed


def get_dataset_from_mltable(
        ws: Workspace,
        mltable_uri: str
) -> Optional[AbstractDataset]:
    """
    Get dataset from MLTable path

    :param ws: workspace to get dataset from
    :param mltable_uri: Path to mltable in workspace
    :return: AbstractDataset

    """
    dataset = None
    try:
        dataset = AbstractDataset._load(mltable_uri, ws)
    except (UserErrorException, ValueError) as e:
        generic_msg = f"Error in loading the dataset from MLTable. Error: {e}"
        _logger.error(generic_msg)
        raise UserErrorException(generic_msg)
    except Exception as e:
        raise UserErrorException(str(e))
    return dataset


def get_pandas_df_from_streamed_dataset(
    amldataset: AbstractDataset
) -> pd.DataFrame:
    """
    Get the pandas dataframe from the tabular dataset
    :param amldataset: AbstractDataset object
    :return: Pandas Dataframe
    """
    dataset_helper = AmlDatasetHelper(amldataset, ignore_data_errors=True)
    return dataset_helper.images_df


def get_streamed_dataset(mltable_path: str) -> AbstractDataset:
    """
    Get the Tabular dataset given the mltable
    :param mltable_path: Path to the MLTable
    :return: AbstractDataset
    """
    my_run = Run.get_context()
    ws = my_run.experiment.workspace
    return get_dataset_from_mltable(ws, mltable_path)


def load_mltable(mltable_path: str, dataset_type: str) -> pd.DataFrame:
    _logger.info("Loading MLTable: {0}".format(mltable_path))
    df: pd.DataFrame = None
    try:
        if dataset_type == DatasetStoreType.PRIVATE:
            # datasets from private datastore needs to be loaded via
            # azureml-dataprep package
            dataset = get_streamed_dataset(mltable_path)
            df = get_pandas_df_from_streamed_dataset(dataset)
        else:
            tbl = mltable.load(mltable_path)
            df: pd.DataFrame = tbl.to_pandas_dataframe()
    except Exception as e:
        _logger.info("Failed to load MLTable")
        _logger.info(e)
    return df


def load_parquet(parquet_path: str) -> pd.DataFrame:
    _logger.info(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return df


def load_dataset(dataset_path: str, dataset_type: str) -> pd.DataFrame:
    _logger.info(f"Attempting to load: {dataset_path}")
    df = load_mltable(dataset_path, dataset_type)
    if df is None:
        df = load_parquet(dataset_path)
    print(df.dtypes)
    print(df.head(10))
    return df


def fetch_model_id(model_info_path: str):
    model_info_path = os.path.join(
        model_info_path,
        DashboardInfo.MODEL_INFO_FILENAME
    )
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
        PropertyKeyValues.RAI_INSIGHTS_VISION_VERSION_KEY:
            responsibleai_vision_version,
        PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY:
            dashboard_info[DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY],
        PropertyKeyValues.RAI_INSIGHTS_DATA_TYPE_KEY:
            IMAGE_DATA_TYPE
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

    # Load training dataset
    _logger.info(f"Dataset type is {args.dataset_type}")
    train_df = load_dataset(args.train_dataset, args.dataset_type)

    # Load evaluation dataset
    _logger.info("Dealing with evaluation dataset")
    test_df = load_dataset(args.test_dataset, args.dataset_type)

    if args.model_input is None or args.model_info is None:
        raise ValueError(
            "Both model info and model input need to be supplied."
        )

    model_id = args.model_info
    _logger.info(f"Loading model: {model_id}")

    my_run = Run.get_context()
    workspace = my_run.experiment.workspace

    model_serializer = ModelSerializer(
        model_id, workspace=workspace,
        model_type=args.model_type,
        use_model_dependency=args.use_model_dependency
    )
    image_model = model_serializer.load("Ignored path")

    if args.task_type == TaskType.MULTILABEL_IMAGE_CLASSIFICATION:
        target_column = get_from_args(
            args=args,
            arg_name="target_column_name",
            custom_parser=json_empty_is_none_parser,
            allow_none=True
        )
    else:
        target_column = args.target_column_name

    _logger.info("Creating RAI Vision Insights")
    rai_vi = RAIVisionInsights(
        model=image_model,
        train=train_df,
        test=test_df,
        target_column=target_column,
        task_type=args.task_type,
        classes=get_from_args(
            args=args,
            arg_name="classes",
            custom_parser=json_empty_is_none_parser,
            allow_none=True
        ),
        maximum_rows_for_test=args.maximum_rows_for_test_dataset,
        serializer=model_serializer
    )

    included_tools: Dict[str, bool] = {
        RAIToolType.CAUSAL: False,
        RAIToolType.COUNTERFACTUAL: False,
        RAIToolType.ERROR_ANALYSIS: False,
        RAIToolType.EXPLANATION: False,
    }

    if args.precompute_explanation:
        _logger.info("Adding explanation")
        rai_vi.explainer.add()
        included_tools[RAIToolType.EXPLANATION] = True

    _logger.info("Starting computation")
    # get keyword arguments
    automl_xai_args = dict()
    for xai_arg in ExplainabilityLiterals.XAI_ARGS_GROUP:
        xai_arg_value = getattr(args, xai_arg)
        if xai_arg_value is not None:
            automl_xai_args[xai_arg] = xai_arg_value
        _logger.info(f"xai_arg: {xai_arg}, value: {xai_arg_value}")

    _logger.info(f"automl_xai_args: {automl_xai_args}")

    rai_vi.compute(**automl_xai_args)
    _logger.info("Computation complete")

    rai_vi.save(args.dashboard)
    _logger.info("Saved dashboard to output")

    rai_data = rai_vi.get_data()
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

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")

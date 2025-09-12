# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import logging
import os
import shutil

import mlflow
from arg_helpers import boolean_parser, json_empty_is_none_parser
from rai_component_utilities import ensure_shim

ensure_shim()
from azureml.rai.utils.telemetry import LoggerFactory, track  # noqa: E402
from constants import (COMPONENT_NAME, DashboardInfo,  # noqa: E402
                       PropertyKeyValues)
from rai_component_utilities import default_json_handler  # noqa: E402
from rai_component_utilities import (fetch_model_id, get_arg,  # noqa: E402
                                     load_dataset, load_mlflow_model)
from raiutils.exceptions import UserConfigValidationException  # noqa: E402
from responsibleai.feature_metadata import FeatureMetadata  # noqa: E402

from responsibleai import RAIInsights  # noqa: E402

DEFAULT_MODULE_NAME = "rai_create_insights"
DEFAULT_MODULE_VERSION = "0.0.0"

_logger = logging.getLogger(__file__)
_ai_logger = None
_module_name = DEFAULT_MODULE_NAME
_module_version = DEFAULT_MODULE_VERSION


def _get_logger():
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = LoggerFactory.get_logger(
            __file__, _module_name, _module_version, COMPONENT_NAME)
    return _ai_logger


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--title", type=str, required=True)

    parser.add_argument(
        "--task_type", type=str, required=True, choices=["classification", "regression", "forecasting"]
    )

    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)

    parser.add_argument("--target_column_name", type=str, required=True)

    parser.add_argument("--maximum_rows_for_test_dataset", type=int, default=5000)

    parser.add_argument(
        "--categorical_column_names", type=str, help="Optional[List[str]]"
    )

    parser.add_argument("--model_info_path", type=str, help="name:version")

    parser.add_argument("--model_input", type=str, help="model local path on remote")

    parser.add_argument("--model_info", type=str, help="name:version")

    parser.add_argument("--classes", type=str, help="Optional[List[str]]")

    parser.add_argument(
        "--feature_metadata",
        type=str,
        help="identity_feature_name:Optional[str], "
             "dropped_features:Optional[List[str]], "
             "datetime_features:Optional[List[str]], "
             "time_series_id_features:Optional[List[str]]",
    )

    parser.add_argument(
        "--use_model_dependency", type=boolean_parser, help="Use model dependency"
    )

    parser.add_argument("--output_path", type=str, help="Path to output JSON")

    # Component info
    parser.add_argument("--component_name", type=str, required=True)
    parser.add_argument("--component_version", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def create_constructor_arg_dict(args):
    """Create a kwarg dict for RAIInsights constructor

    Only does the 'parameters' for the component, not the
    input ports
    """
    result = dict()

    cat_col_names = get_arg(
        args, "categorical_column_names", custom_parser=json.loads, allow_none=True
    )
    class_names = get_arg(
        args, "classes", custom_parser=json_empty_is_none_parser, allow_none=True
    )
    feature_metadata_dict = get_arg(
        args, "feature_metadata", custom_parser=json.loads, allow_none=True
    )
    feature_metadata = FeatureMetadata()
    try:
        for key in [
                PropertyKeyValues.RAI_INSIGHTS_DROPPED_FEATURE_KEY,
                PropertyKeyValues.RAI_INSIGHTS_IDENTITY_FEATURE_KEY,
                PropertyKeyValues.RAI_INSIGHTS_DATETIME_FEATURES_KEY,
                PropertyKeyValues.RAI_INSIGHTS_TIME_SERIES_ID_FEATURES_KEY]:
            if key in feature_metadata_dict.keys():
                setattr(feature_metadata, key, feature_metadata_dict[key])
    except AttributeError as e:
        raise UserConfigValidationException(f"Feature metadata should be parsed to a dictionary. {e}")

    if cat_col_names:
        feature_metadata.categorical_features = cat_col_names

    result["target_column"] = args.target_column_name
    result["task_type"] = args.task_type
    result["classes"] = class_names
    result["feature_metadata"] = feature_metadata
    result["maximum_rows_for_test"] = args.maximum_rows_for_test_dataset

    return result


def copy_input_data(component_input_path: str, output_path: str):
    # asset id
    if component_input_path.lower().startswith("azureml://"):
        output_file = os.path.join(output_path, "assetid")
        os.makedirs(output_path, exist_ok=True)
        with open(output_file, "w") as of:
            of.write(component_input_path)
        return

    if os.path.isdir(component_input_path):
        src_path = component_input_path
    else:
        src_path = os.path.dirname(component_input_path)
    src_path = src_path + "/"
    _logger.info(f"Copying from {src_path} to {output_path}")
    assert os.path.isdir(src_path), "Checking src_path"
    shutil.copytree(src=src_path, dst=output_path)


@track(_get_logger)
def main(args):
    # Get MLflow run ID
    current_run = mlflow.active_run()
    if current_run is not None:
        run_id = current_run.info.run_id
    else:
        # Fallback to starting a new run if no active run
        with mlflow.start_run() as run:
            run_id = run.info.run_id

    _logger.info("Dealing with initialization dataset")
    train_df = load_dataset(args.train_dataset)

    _logger.info("Dealing with evaluation dataset")
    test_df = load_dataset(args.test_dataset)

    if args.model_info_path is None and (
        args.model_input is None or args.model_info is None
    ):
        raise UserConfigValidationException("Either model info or model input needs to be supplied.")

    model_estimator = None
    model_id = None
    # For now, the separate conda env will only be used for forecasting.
    # At a later point, we might enable this for all task types.
    use_separate_conda_env = args.task_type == "forecasting"
    if args.model_info_path:
        model_id = fetch_model_id(args.model_info_path)
        _logger.info("Loading model from model_info_path: {0}".format(model_id))
        model_estimator = load_mlflow_model(
            use_model_dependency=args.use_model_dependency,
            use_separate_conda_env=use_separate_conda_env,
            model_id=model_id,
        )
    elif args.model_input and args.model_info:
        model_id = args.model_info
        _logger.info("Loading model from model_info: {0}".format(model_id))
        model_estimator = load_mlflow_model(
            use_model_dependency=args.use_model_dependency,
            use_separate_conda_env=use_separate_conda_env,
            model_path=args.model_input,
        )

    # unwrap the model if it's an sklearn wrapper
    if model_estimator.__class__.__name__ == "_SklearnModelWrapper":
        model_estimator = model_estimator.sklearn_model

    constructor_args = create_constructor_arg_dict(args)

    # Make sure that it actually loads
    _logger.info("Creating RAIInsights object")
    _ = RAIInsights(
        model=model_estimator, train=train_df, test=test_df, forecasting_enabled=True, **constructor_args
    )

    _logger.info("Saving JSON for tool components")
    output_dict = {
        DashboardInfo.RAI_INSIGHTS_RUN_ID_KEY: run_id,
        DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY: model_id,
        DashboardInfo.RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY: constructor_args,
        DashboardInfo.RAI_INSIGHTS_TRAIN_DATASET_ID_KEY: os.path.basename(args.train_dataset),
        DashboardInfo.RAI_INSIGHTS_TEST_DATASET_ID_KEY: os.path.basename(args.test_dataset),
        DashboardInfo.RAI_INSIGHTS_DASHBOARD_TITLE_KEY: args.title,
        DashboardInfo.RAI_INSIGHTS_INPUT_ARGS_KEY: vars(args),
    }
    output_file = os.path.join(
        args.output_path, DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME
    )
    with open(output_file, "w") as of:
        json.dump(output_dict, of, default=default_json_handler)

    _logger.info("Copying train data files")
    copy_input_data(
        args.train_dataset,
        os.path.join(args.output_path, DashboardInfo.TRAIN_FILES_DIR),
    )
    _logger.info("Copying test data files")
    copy_input_data(
        args.test_dataset, os.path.join(args.output_path, DashboardInfo.TEST_FILES_DIR)
    )


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

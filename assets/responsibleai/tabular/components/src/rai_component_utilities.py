# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mltable
import pandas as pd
import requests
from arg_helpers import get_from_args
from azureml.core import Model, Run, Workspace
# TODO: seems this method needs to be made public
from azureml.rai.utils.telemetry.loggerfactory import _extract_and_filter_stack
from constants import (MLFLOW_MODEL_SERVER_PORT, DashboardInfo,
                       PropertyKeyValues, RAIToolType)
from raiutils.exceptions import UserConfigValidationException
from responsibleai._internal._served_model_wrapper import ServedModelWrapper
from responsibleai.feature_metadata import FeatureMetadata

from responsibleai import RAIInsights
from responsibleai import __version__ as responsibleai_version

assetid_re = re.compile(
    r"azureml://locations/(?P<location>.*)/workspaces/(?P<workspaceid>.*)/(?P<assettype>.*)/(?P<assetname>.*)/versions/(?P<assetversion>.*)"  # noqa: E501
)
data_type = "data_type"

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


# Directory names saved by RAIInsights might not match tool names
_tool_directory_mapping: Dict[str, str] = {
    RAIToolType.CAUSAL: "causal",
    RAIToolType.COUNTERFACTUAL: "counterfactual",
    RAIToolType.ERROR_ANALYSIS: "error_analysis",
    RAIToolType.EXPLANATION: "explainer",
}


class UserConfigError(Exception):
    def __init__(self, message, cause=None):
        if cause:
            self.tb = _extract_and_filter_stack(cause, traceback.extract_tb(sys.exc_info()[2]))
            self.cause = cause
        super().__init__(message)


def print_dir_tree(base_dir):
    print("\nBEGIN DIRTREE")
    for current_dir, subdirs, files in os.walk(base_dir):
        # Current Iteration Directory
        print(current_dir)

        # Directories
        for dirname in sorted(subdirs):
            print("\t" + dirname + "/")

        # Files
        for filename in sorted(files):
            print("\t" + filename)
    print("END DIRTREE\n", flush=True)


def fetch_model_id(model_info_path: str):
    model_info_path = os.path.join(model_info_path, DashboardInfo.MODEL_INFO_FILENAME)
    try:
        json_file = open(model_info_path, "r")
    except Exception:
        raise UserConfigValidationException(
            f"Failed to open {model_info_path}. Please ensure the model path is correct."
        )
    model_info = json.load(json_file)
    json_file.close()
    if DashboardInfo.MODEL_ID_KEY not in model_info:
        raise UserConfigValidationException(
            f"Invalid input, expecting key {DashboardInfo.MODEL_ID_KEY} to exist in the input json"
        )
    else:
        return model_info[DashboardInfo.MODEL_ID_KEY]


def load_mlflow_model(
    workspace: Workspace,
    use_model_dependency: bool = False,
    use_separate_conda_env: bool = False,
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Any:
    model_uri = model_path
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())

    if model_id:
        try:
            model = Model._get(workspace, id=model_id)
        except Exception as e:
            raise UserConfigError(
                "Unable to retrieve model by model id {} in workspace {}, error:\n{}".format(
                    model_id, workspace.name, e
                ), e
            )
        model_uri = "models:/{}/{}".format(model.name, model.version)

    if use_model_dependency:
        if not use_separate_conda_env:
            try:
                conda_file = mlflow.pyfunc.get_model_dependencies(model_uri, format="conda")
            except Exception as e:
                raise UserConfigError(
                    "Failed to get model dependency from given model {}, error:\n{}".format(
                        model_uri, e
                    ), e
                )
        try:
            if use_separate_conda_env:
                conda_install_command = ["mlflow", "models", "prepare-env",
                                         "-m", model_uri,
                                         "--env-manager", "conda"]
            else:
                # mlflow model input mount as read only. Conda need write access.
                local_conda_dep = "./conda_dep.yaml"
                shutil.copyfile(conda_file, local_conda_dep)
                conda_prefix = str(Path(sys.executable).parents[1])
                conda_install_command = ["conda", "env", "update",
                                         "--prefix", conda_prefix,
                                         "-f", local_conda_dep]

            install_log = subprocess.check_output(conda_install_command)
            _logger.info(
                "Conda dependency installation successful, logs: {}".format(
                    install_log
                )
            )
        except subprocess.CalledProcessError as e:
            _logger.error(
                "Installing dependency using conda.yaml from mlflow model failed: {}".format(
                    e.output
                )
            )
            _classify_and_log_pip_install_error(e.output)
            raise UserConfigValidationException(
                "Installing dependency using conda environment spec from mlflow model failed. "
                "This behavior can be turned off with setting use_model_dependency to False in job spec. "
                "You may also check error log above to manually resolve package conflict error"
            )
        _logger.info("Successfully installed model dependencies")

    try:
        if not use_separate_conda_env:
            model = mlflow.pyfunc.load_model(model_uri)._model_impl
            return model

        # Serve model from separate conda env using mlflow
        mlflow_models_serve_logfile_name = "mlflow_models_serve.log"
        try:
            # run mlflow model server in background
            with open(mlflow_models_serve_logfile_name, "w") as logfile:
                model_serving_log = subprocess.Popen(
                    [
                        "mlflow",
                        "models",
                        "serve",
                        "-m",
                        model_uri,
                        "--env-manager",
                        "conda",
                        "-p",
                        str(MLFLOW_MODEL_SERVER_PORT)
                    ],
                    close_fds=True,
                    stdout=logfile,
                    stderr=logfile
                )
            _logger.info("Started mlflow model server process in the background")
        except subprocess.CalledProcessError as e:
            _logger.error(
                f"Running mlflow models serve in the background failed: {e.output}"
            )
            _classify_and_log_pip_install_error(e.output)
            raise RuntimeError(
                "Starting the mlflow model server failed."
            )

        # If the server started successfully then the logfile should contain a line
        # saying "Listening at: http"
        # If not, it could either take more time to start or it failed to start.
        # We can check if the process ended by calling poll() on it.
        # Otherwise, we wait a predefined time and check again.
        for _ in range(10):
            with open(mlflow_models_serve_logfile_name, "r") as logfile:
                logs = logfile.read()
                if "Listening at: http" not in logs:
                    if model_serving_log.poll() is not None:
                        # process ended
                        raise RuntimeError(
                            f"Unable to start mlflow model server: {logs}"
                        )
                    # process still running, wait and try again...
                else:
                    try:
                        # attempt to contact mlflow model server
                        # if the response is a 500 (due to missing body) then the server is up
                        # if it's a 404 then the server is just starting up and we need to wait
                        test_response = requests.post(f"http://localhost:{MLFLOW_MODEL_SERVER_PORT}/invocations")
                        if test_response.status_code == 500:
                            break

                    except Exception as e:
                        _logger.info(
                            "Waiting for mlflow model server to start, error: {}".format(
                                e
                            )
                        )
            time.sleep(5)
        else:
            raise RuntimeError(
                "Unable to start mlflow model server."
            )
        _logger.info("Successfully started mlflow model server.")
        model = ServedModelWrapper(port=MLFLOW_MODEL_SERVER_PORT)
        return model
    except Exception as e:
        raise UserConfigError(
            "Unable to load mlflow model from {} in current environment due to error:\n{}".format(
                model_uri, e
            ), e
        )


def _classify_and_log_pip_install_error(elog):
    ret_message = []
    if elog is None:
        return ret_message

    if b"Could not find a version that satisfies the requirement" in elog:
        ret_message.append("Detected unsatisfiable version requirment.")

    if b"package versions have conflicting dependencies" in elog:
        ret_message.append("Detected dependency conflict error.")

    for m in ret_message:
        _logger.warning(m)

    return ret_message


def load_mltable(mltable_path: str) -> pd.DataFrame:
    _logger.info(f"Attempting to load {mltable_path} as MLTable")
    try:
        assetid_path = os.path.join(mltable_path, "assetid")
        if os.path.exists(assetid_path):
            with open(assetid_path, "r") as assetid_file:
                mltable_path = assetid_file.read()

        tbl = mltable.load(mltable_path)
        df: pd.DataFrame = tbl.to_pandas_dataframe()
    except Exception as e:
        _logger.info(f"Failed to load {mltable_path} as MLTable. ")
        raise e
    return df


def load_parquet(parquet_path: str) -> pd.DataFrame:
    _logger.info(f"Attempting to load {parquet_path} as parquet dataset")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        _logger.info(f"Failed to load {parquet_path} as MLTable. ")
        raise e
    return df


def load_dataset(dataset_path: str) -> pd.DataFrame:
    _logger.info(f"Attempting to load: {dataset_path}")
    exceptions = []
    isLoadSuccessful = False

    try:
        df = load_mltable(dataset_path)
        isLoadSuccessful = True
    except Exception as e:
        new_e = UserConfigError(
            f"Input dataset {dataset_path} cannot be read as mltable."
            f"You may disregard this error if dataset input is intended to be parquet dataset. Exception: {e}",
            e
        )
        exceptions.append(new_e)

    if not isLoadSuccessful:
        try:
            df = load_parquet(dataset_path)
            isLoadSuccessful = True
        except Exception as e:
            new_e = UserConfigError(
                f"Input dataset {dataset_path} cannot be read as parquet."
                f"You may disregard this error if dataset input is intended to be mltable. Exception: {e}",
                e
            )
            exceptions.append(new_e)

    if not isLoadSuccessful:
        raise UserConfigError(
            f"Input dataset {dataset_path} cannot be read as MLTable or Parquet dataset."
            f"Please check that input dataset is valid. Exceptions encountered during reading: {exceptions}"
        )

    print(df.dtypes)
    print(df.head(10))
    return df


def load_dashboard_info_file(input_port_path: str) -> Dict[str, str]:
    # Load the rai_insights_dashboard file info
    rai_insights_dashboard_file = os.path.join(
        input_port_path, DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME
    )
    with open(rai_insights_dashboard_file, "r") as si:
        dashboard_info = json.load(si, object_hook=default_object_hook)
    _logger.info("rai_insights_parent info: {0}".format(dashboard_info))
    return dashboard_info


def copy_dashboard_info_file(src_port_path: str, dst_port_path: str):
    src = pathlib.Path(src_port_path) / DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME
    dst = pathlib.Path(dst_port_path) / DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME

    shutil.copyfile(src, dst)


def create_rai_tool_directories(rai_insights_dir: pathlib.Path) -> None:
    # Have to create empty subdirectories for the managers
    # THe RAI Insights object expect these to be present, but
    # since directories don't actually exist in Azure Blob store
    # they may not be present (some of the tools always have
    # a file present, even if no tool instances have been added)
    for v in _tool_directory_mapping.values():
        os.makedirs(rai_insights_dir / v, exist_ok=True)
    _logger.info("Added empty directories")


def load_rai_insights_from_input_port(input_port_path: str) -> RAIInsights:
    with tempfile.TemporaryDirectory() as incoming_temp_dir:
        incoming_dir = pathlib.Path(incoming_temp_dir)
        shutil.copytree(input_port_path, incoming_dir, dirs_exist_ok=True)
        _logger.info("Copied RAI Insights input to temporary directory")

        create_rai_tool_directories(incoming_dir)

        result = RAIInsights.load(incoming_dir)
        _logger.info("Loaded RAIInsights object")
    return result


def copy_insight_to_raiinsights(
    rai_insights_dir: pathlib.Path, insight_dir: pathlib.Path
) -> str:
    print("Starting copy")

    # Recall that we copy the JSON containing metadata from the
    # constructor component into each directory
    # This means we have that file and the results directory
    # present in the insight_dir
    dir_items = list(insight_dir.iterdir())
    assert len(dir_items) == 2

    # We want the directory, not the JSON file
    if dir_items[0].name == DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME:
        tool_dir_name = dir_items[1].name
    else:
        tool_dir_name = dir_items[0].name

    _logger.info("Detected tool: {0}".format(tool_dir_name))
    assert tool_dir_name in _tool_directory_mapping.values()
    for k, v in _tool_directory_mapping.items():
        if tool_dir_name == v:
            tool_type = k
    _logger.info("Mapped to tool: {0}".format(tool_type))
    tool_dir = insight_dir / tool_dir_name

    tool_dir_items = list(tool_dir.iterdir())
    assert len(tool_dir_items) == 1

    if tool_type == RAIToolType.EXPLANATION:
        # Explanations will have a directory already present for some reason
        # Furthermore we only support one explanation per dashboard for
        # some other reason
        # Put together, if we have an explanation, we need to remove
        # what's there already or we can get confused
        _logger.info("Detected explanation, removing existing directory")
        for item in (rai_insights_dir / tool_dir_name).iterdir():
            _logger.info("Removing directory {0}".format(str(item)))
            shutil.rmtree(item)

    src_dir = insight_dir / tool_dir_name / tool_dir_items[0].parts[-1]
    dst_dir = rai_insights_dir / tool_dir_name / tool_dir_items[0].parts[-1]
    shutil.copytree(
        src=src_dir,
        dst=dst_dir,
    )

    _logger.info("Copy complete")
    return tool_type


def save_to_output_port(rai_i: RAIInsights, output_port_path: str, tool_type: str):
    with tempfile.TemporaryDirectory() as tmpdirname:
        rai_i.save(tmpdirname)
        _logger.info(f"Saved to {tmpdirname}")

        tool_dir_name = _tool_directory_mapping[tool_type]
        insight_dirs = os.listdir(pathlib.Path(tmpdirname) / tool_dir_name)
        assert len(insight_dirs) == 1, "Checking for exactly one tool output"
        _logger.info("Checking dirname is GUID")
        uuid.UUID(insight_dirs[0])

        target_path = pathlib.Path(output_port_path) / tool_dir_name
        target_path.mkdir()
        _logger.info("Created output directory")

        _logger.info("Starting copy")
        shutil.copytree(
            pathlib.Path(tmpdirname) / tool_dir_name,
            target_path,
            dirs_exist_ok=True,
        )
    _logger.info("Copied to output")


def add_properties_to_gather_run(
    dashboard_info: Dict[str, str], tool_present_dict: Dict[str, str]
):
    _logger.info("Adding properties to the gather run")
    gather_run = Run.get_context()

    run_properties = {
        PropertyKeyValues.RAI_INSIGHTS_TYPE_KEY: PropertyKeyValues.RAI_INSIGHTS_TYPE_GATHER,
        PropertyKeyValues.RAI_INSIGHTS_DASHBOARD_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_RUN_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY: responsibleai_version,
        PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_TEST_DATASET_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_TEST_DATASET_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_TRAIN_DATASET_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_TRAIN_DATASET_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_DASHBOARD_TITLE_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_DASHBOARD_TITLE_KEY
        ],
    }

    _logger.info("Appending tool present information")
    for k, v in tool_present_dict.items():
        key = PropertyKeyValues.RAI_INSIGHTS_TOOL_KEY_FORMAT.format(k)
        run_properties[key] = str(v)

    _logger.info("Making service call")
    gather_run.add_properties(run_properties)
    _logger.info("Properties added to gather run")


def create_rai_insights_from_port_path(my_run: Run, port_path: str) -> RAIInsights:
    _logger.info("Creating RAIInsights from constructor component output")

    _logger.info("Loading data files")
    df_train = load_dataset(os.path.join(port_path, DashboardInfo.TRAIN_FILES_DIR))
    df_test = load_dataset(os.path.join(port_path, DashboardInfo.TEST_FILES_DIR))

    _logger.info("Loading config file")
    config = load_dashboard_info_file(port_path)
    constructor_args = config[DashboardInfo.RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY]
    _logger.info(f"Constuctor args: {constructor_args}")

    _logger.info("Loading model")
    input_args = config[DashboardInfo.RAI_INSIGHTS_INPUT_ARGS_KEY]
    use_model_dependency = input_args["use_model_dependency"]
    model_id = config[DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY]
    _logger.info("Loading model: {0}".format(model_id))

    # For now, the separate conda env will only be used for forecasting.
    # At a later point, we might enable this for all task types.
    use_separate_conda_env = False
    if "task_type" in constructor_args:
        is_forecasting_task = constructor_args["task_type"] == "forecasting"
        use_separate_conda_env = is_forecasting_task
        constructor_args["forecasting_enabled"] = is_forecasting_task

    model_estimator = load_mlflow_model(
        workspace=my_run.experiment.workspace,
        use_model_dependency=use_model_dependency,
        use_separate_conda_env=use_separate_conda_env,
        model_id=model_id,
    )

    # unwrap the model if it's an sklearn wrapper
    if model_estimator.__class__.__name__ == "_SklearnModelWrapper":
        model_estimator = model_estimator.sklearn_model

    _logger.info("Creating RAIInsights object")
    rai_i = RAIInsights(
        model=model_estimator, train=df_train, test=df_test, **constructor_args
    )
    return rai_i


def get_run_input_assets(run):
    return run.get_details()["runDefinition"]["inputAssets"]


def get_asset_information(assetid):
    match = assetid_re.match(assetid)

    return match.groupdict()


def get_train_dataset_id(run):
    return get_dataset_name_version(run, "train_dataset")


def get_test_dataset_id(run):
    return get_dataset_name_version(run, "test_dataset")


def get_dataset_name_version(run, dataset_input_name):
    aid = get_run_input_assets(run)[dataset_input_name]["asset"]["assetId"]
    ainfo = get_asset_information(aid)
    return f'{ainfo["assetname"]}:{ainfo["assetversion"]}'


def default_json_handler(data):
    if isinstance(data, FeatureMetadata):
        meta_dict = data.__dict__
        type_name = type(data).__name__
        meta_dict[data_type] = type_name
        return meta_dict
    return None


def default_object_hook(dict):
    if data_type in dict and dict[data_type] == FeatureMetadata.__name__:
        del dict[data_type]
        return FeatureMetadata(**dict)
    return dict


def get_arg(args, arg_name: str, custom_parser, allow_none: bool) -> Any:
    try:
        return get_from_args(args, arg_name, custom_parser, allow_none)
    except ValueError as e:
        raise UserConfigError(
            f"Unable to parse {arg_name} from {args}."
            f"Please check that {args} is valid input and that {arg_name} exists."
            "For example, a json string with unquoted string value or key can cause this error."
            f"Raw parsing error: {e}"
        )

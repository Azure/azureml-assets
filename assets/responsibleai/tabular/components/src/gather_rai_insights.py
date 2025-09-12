# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
from rai_component_utilities import ensure_shim

ensure_shim()
from azureml.rai.utils.telemetry import LoggerFactory, track  # noqa: E402
from constants import (COMPONENT_NAME, MLFLOW_MODEL_SERVER_PORT,  # noqa: E402
                       DashboardInfo, PropertyKeyValues, RAIToolType)
from rai_component_utilities import add_properties_to_gather_run  # noqa: E402
from rai_component_utilities import copy_insight_to_raiinsights  # noqa: E402
from rai_component_utilities import \
    create_rai_insights_from_port_path  # noqa: E402
from rai_component_utilities import create_rai_tool_directories  # noqa: E402
from rai_component_utilities import default_json_handler  # noqa: E402
from rai_component_utilities import load_dashboard_info_file  # noqa: E402
from rai_component_utilities import print_dir_tree  # noqa: E402
from responsibleai.serialization_utilities import \
    serialize_json_safe  # noqa: E402

from responsibleai import RAIInsights  # noqa: E402
from responsibleai import __version__ as responsibleai_version  # noqa: E402

_DASHBOARD_CONSTRUCTOR_MISMATCH = (
    "Insight {0} was not " "computed from the constructor specified"
)
_DUPLICATE_TOOL = "Insight {0} is of type {1} which is already present"

DEFAULT_MODULE_NAME = "rai_gather_insights"
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


def add_properties_to_gather_run_rai_insights(
    dashboard_info: Dict[str, str], tool_present_dict: Dict[str, str]
):
    """Local wrapper for the common add_properties_to_gather_run function."""
    run_properties = {
        PropertyKeyValues.RAI_INSIGHTS_TYPE_KEY: PropertyKeyValues.RAI_INSIGHTS_TYPE_GATHER,
        PropertyKeyValues.RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY: responsibleai_version,
        PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_DASHBOARD_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_RUN_ID_KEY
        ],
    }

    # Call the common function
    add_properties_to_gather_run(
        dashboard_info, run_properties, tool_present_dict, _module_name, _module_version
    )


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--constructor", type=str, required=True)
    parser.add_argument("--insight_1", type=str, default=None)
    parser.add_argument("--insight_2", type=str, default=None)
    parser.add_argument("--insight_3", type=str, default=None)
    parser.add_argument("--insight_4", type=str, default=None)
    parser.add_argument("--dashboard", type=str, required=True)
    parser.add_argument("--ux_json", type=str, required=True)

    # Component info
    parser.add_argument("--component_name", type=str, required=True)
    parser.add_argument("--component_version", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


@track(_get_logger)
def main(args):
    dashboard_info = load_dashboard_info_file(args.constructor)
    _logger.info("Constructor info: {0}".format(dashboard_info))

    with tempfile.TemporaryDirectory() as incoming_temp_dir:
        incoming_dir = Path(incoming_temp_dir)
        rai_temp = create_rai_insights_from_port_path(args.constructor)

        # We need to fix the following issue in RAIInsights.
        # When using the served model wrapper, predictions (currently
        # forecasts only) are stored as lists instead of numpy.ndarray.
        # This causes the following error when saving the RAIInsights object:
        # AttributeError: 'list' object has no attribute 'tolist'
        # To remedy this, the code below converts the lists to numpy.ndarray.
        # Once it is fixed in RAIInsights, this code can be removed.
        for method_name in ['forecast', 'forecast_quantiles']:
            field_name = f"_{method_name}_output"
            if field_name in rai_temp.__dict__ and \
                    isinstance(rai_temp._forecast_output, list):
                setattr(rai_temp, field_name, np.array(rai_temp._forecast_output))

        rai_temp.save(incoming_temp_dir)

        print("Saved rai_temp")
        print_dir_tree(incoming_temp_dir)
        print("=======")

        create_rai_tool_directories(incoming_dir)
        _logger.info("Saved empty RAI Insights input to temporary directory")

        insight_paths = [
            args.insight_1,
            args.insight_2,
            args.insight_3,
            args.insight_4,
        ]

        included_tools: Dict[str, bool] = {
            RAIToolType.CAUSAL: False,
            RAIToolType.COUNTERFACTUAL: False,
            RAIToolType.ERROR_ANALYSIS: False,
            RAIToolType.EXPLANATION: False,
        }
        for i in range(len(insight_paths)):
            current_insight_arg = insight_paths[i]
            if current_insight_arg is not None:
                current_insight_path = Path(current_insight_arg)
                _logger.info("Checking dashboard info")
                insight_info = load_dashboard_info_file(current_insight_path)
                _logger.info("Insight info: {0}".format(insight_info))

                # Cross check against the constructor
                if insight_info != dashboard_info:
                    err_string = _DASHBOARD_CONSTRUCTOR_MISMATCH.format(i + 1)
                    raise ValueError(err_string)

                # Copy the data
                _logger.info("Copying insight {0}".format(i + 1))
                tool = copy_insight_to_raiinsights(incoming_dir, current_insight_path)

                # Can only have one instance of each tool
                if included_tools[tool]:
                    err_string = _DUPLICATE_TOOL.format(i + 1, tool)
                    raise ValueError(err_string)

                included_tools[tool] = True
            else:
                _logger.info("Insight {0} is None".format(i + 1))

        _logger.info("Tool summary: {0}".format(included_tools))

        if rai_temp.task_type == "forecasting":
            # Set model serving port to arbitrary value to avoid loading the
            # model.
            # Forecasting uses the ServedModelWrapper which cannot be
            # (de-)serialized like other models.
            # Note that the port number is selected arbitrarily.
            # The model is not actually being served.
            os.environ["RAI_MODEL_SERVING_PORT"] = str(MLFLOW_MODEL_SERVER_PORT)

        rai_i = RAIInsights.load(incoming_dir)
        _logger.info("Object loaded")

        rai_i.save(args.dashboard)

        # Get MLflow run ID
        current_run = mlflow.active_run()
        if current_run is not None:
            run_id = current_run.info.run_id
        else:
            # Fallback to starting a new run if no active run
            with mlflow.start_run() as run:
                run_id = run.info.run_id

        # update dashboard info with gather job run id and write
        dashboard_info[DashboardInfo.RAI_INSIGHTS_GATHER_RUN_ID_KEY] = run_id
        output_file = os.path.join(
            args.dashboard, DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME
        )
        with open(output_file, "w") as of:
            json.dump(dashboard_info, of, default=default_json_handler)

        _logger.info("Saved dashboard to output")

        rai_data = rai_i.get_data()
        rai_dict = serialize_json_safe(rai_data)
        json_filename = "dashboard.json"
        output_path = Path(args.ux_json) / json_filename
        with open(output_path, "w") as json_file:
            json.dump(rai_dict, json_file)
        _logger.info("Dashboard JSON written")

        add_properties_to_gather_run_rai_insights(dashboard_info, included_tools)
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

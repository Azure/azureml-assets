# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging

from rai_component_utilities import ensure_shim

ensure_shim()
from azureml.rai.utils.telemetry import LoggerFactory, track  # noqa: E402
from constants import COMPONENT_NAME, RAIToolType  # noqa: E402
from rai_component_utilities import copy_dashboard_info_file  # noqa: E402
from rai_component_utilities import \
    create_rai_insights_from_port_path  # noqa: E402
from rai_component_utilities import save_to_output_port  # noqa: E402

from responsibleai import RAIInsights  # noqa: E402

DEFAULT_MODULE_NAME = "rai_explanation_insights"
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

    parser.add_argument("--rai_insights_dashboard", type=str, required=True)
    parser.add_argument("--comment", type=str, required=True)
    parser.add_argument("--explanation_path", type=str, required=True)

    # Component info
    parser.add_argument("--component_name", type=str, required=True)
    parser.add_argument("--component_version", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


@track(_get_logger)
def main(args):
    # Create the RAI Insights object
    rai_i: RAIInsights = create_rai_insights_from_port_path(
        args.rai_insights_dashboard
    )

    # Add the explanation
    rai_i.explainer.add()
    _logger.info("Added explanation")

    # Compute
    rai_i.compute()
    _logger.info("Computation complete")

    # Save
    save_to_output_port(rai_i, args.explanation_path, RAIToolType.EXPLANATION)
    _logger.info("Saved to output port")

    # Copy the dashboard info file
    copy_dashboard_info_file(args.rai_insights_dashboard, args.explanation_path)

    _logger.info("Completing")


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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging

from arg_helpers import (boolean_parser, json_empty_is_none_parser,
                         str_or_int_parser, str_or_list_parser)
from rai_component_utilities import ensure_shim

ensure_shim()
from azureml.rai.utils.telemetry import LoggerFactory, track  # noqa: E402
from constants import COMPONENT_NAME, RAIToolType  # noqa: E402
from rai_component_utilities import copy_dashboard_info_file  # noqa: E402
from rai_component_utilities import (  # noqa: E402
    create_rai_insights_from_port_path, save_to_output_port)

from responsibleai import RAIInsights  # noqa: E402

DEFAULT_MODULE_NAME = "rai_counterfactual_insights"
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
    parser.add_argument("--total_CFs", type=int, required=True)
    parser.add_argument("--method", type=str)
    parser.add_argument("--desired_class", type=str_or_int_parser)
    parser.add_argument("--desired_range", type=json_empty_is_none_parser, help="List")
    parser.add_argument(
        "--permitted_range", type=json_empty_is_none_parser, help="Dict"
    )
    parser.add_argument("--features_to_vary", type=str_or_list_parser)
    parser.add_argument("--feature_importance", type=boolean_parser)
    parser.add_argument("--counterfactual_path", type=str)

    # Component info
    parser.add_argument("--component_name", type=str, required=True)
    parser.add_argument("--component_version", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


@track(_get_logger)
def main(args):
    # Load the RAI Insights object
    rai_i: RAIInsights = create_rai_insights_from_port_path(
        args.rai_insights_dashboard
    )
    # Add the counterfactual
    rai_i.counterfactual.add(
        total_CFs=args.total_CFs,
        method=args.method,
        desired_class=args.desired_class,
        desired_range=args.desired_range,
        permitted_range=args.permitted_range,
        features_to_vary=args.features_to_vary,
        feature_importance=args.feature_importance,
    )
    _logger.info("Added counterfactual")

    # Compute
    rai_i.compute()
    _logger.info("Computation complete")

    # Save
    save_to_output_port(rai_i, args.counterfactual_path, RAIToolType.COUNTERFACTUAL)
    _logger.info("Saved to output port")

    # Copy the dashboard info file
    copy_dashboard_info_file(args.rai_insights_dashboard, args.counterfactual_path)

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

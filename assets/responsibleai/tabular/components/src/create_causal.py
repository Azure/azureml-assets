# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import logging

from arg_helpers import (boolean_parser, float_or_json_parser,
                         int_or_none_parser, str_or_list_parser)
from constants import COMPONENT_NAME, RAIToolType
from rai_component_utilities import (copy_dashboard_info_file,
                                     create_rai_insights_from_port_path,
                                     ensure_shim, save_to_output_port)

from responsibleai import RAIInsights

ensure_shim()
from azureml.rai.utils.telemetry import LoggerFactory, track  # noqa: E402

DEFAULT_MODULE_NAME = "rai_causal_insights"
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

    parser.add_argument("--treatment_features", type=json.loads, help="List[str]")
    parser.add_argument(
        "--heterogeneity_features",
        type=json.loads,
        help="Optional[List[str]] use 'null' to skip",
    )
    parser.add_argument("--nuisance_model", type=str)
    parser.add_argument("--heterogeneity_model", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--upper_bound_on_cat_expansion", type=int)
    parser.add_argument(
        "--treatment_cost",
        type=float_or_json_parser,
        help="Union[float, List[Union[float, np.ndarray]]]",
    )
    parser.add_argument("--min_tree_leaf_samples", type=int)
    parser.add_argument("--max_tree_depth", type=int)
    parser.add_argument("--skip_cat_limit_checks", type=boolean_parser)
    parser.add_argument("--categories", type=str_or_list_parser)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--verbose", type=int)
    parser.add_argument("--random_state", type=int_or_none_parser)

    parser.add_argument("--causal_path", type=str)

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

    # Add the causal analysis
    rai_i.causal.add(
        treatment_features=args.treatment_features,
        heterogeneity_features=args.heterogeneity_features,
        nuisance_model=args.nuisance_model,
        heterogeneity_model=args.heterogeneity_model,
        alpha=args.alpha,
        upper_bound_on_cat_expansion=args.upper_bound_on_cat_expansion,
        treatment_cost=args.treatment_cost,
        min_tree_leaf_samples=args.min_tree_leaf_samples,
        max_tree_depth=args.max_tree_depth,
        skip_cat_limit_checks=args.skip_cat_limit_checks,
        categories=args.categories,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_state=args.random_state,
    )
    _logger.info("Added causal")

    # Compute
    rai_i.compute()
    _logger.info("Computation complete")

    # Save
    save_to_output_port(rai_i, args.causal_path, RAIToolType.CAUSAL)
    _logger.info("Saved computation to output port")

    # Copy the dashboard info file
    copy_dashboard_info_file(args.rai_insights_dashboard, args.causal_path)

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

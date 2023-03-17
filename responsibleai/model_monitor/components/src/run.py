# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Histogram Component."""

import argparse
import logging
from calculate_attribution import compute_attribution_drift
from io_utils import load_mltable_to_df

from tabular.components.src._telemetry._loggerfactory import _LoggerFactory

_logger = logging.getLogger(__file__)
_ai_logger = None


def _get_logger():
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = _LoggerFactory.get_logger(__file__)
    return _ai_logger


_get_logger()


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_data", type=str)
    parser.add_argument("--production_data", type=str)

    args = parser.parse_args()

    return args


def run(args):

    baseline_df = load_mltable_to_df(args.baseline_data)
    production_df = load_mltable_to_df(args.production_data)

    # todo: read from args when available
    task_type = "classification"
    target_column = "target"
    try:
        compute_attribution_drift(task_type, target_column, baseline_df, production_df)
    except Exception as e:
        _logger.info("Error encountered when executing feature attribution component: {0}", e)

    _logger.info("Successfully executed the feature attribution component.")


if __name__ == "__main__":
    args = parse_args()

    run(args)

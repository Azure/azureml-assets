# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate computing RAI insights for AutoML run using responsibleai docker image."""
# imports
import argparse
from responsibleai_tabular_automl import (
    compute_and_upload_rai_insights
)


def main(args):
    compute_and_upload_rai_insights(
        args.automl_parent_run_id,
        args.automl_child_run_id)


def parse_args():
    """Parse arguments."""
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--automl_parent_run_id", type=str)
    parser.add_argument("--automl_child_run_id", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)

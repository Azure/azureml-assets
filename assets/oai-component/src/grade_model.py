import json
import logging
from argparse import ArgumentParser

import pandas as pd

from graders.config.config_reader import get_config
from data.data_reader import get_data
from graders.get_graders import get_grader

logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--predictions_data", type=str, dest="predictions_data", required=True)
    parser.add_argument("--config_str", type=str, dest="config_str", required=True, default=None)
    parser.add_argument("--grader_result", type=str, dest="grader_result")
    parser.add_argument("--ground_truth", type=str, dest="ground_truth")
    parser.add_argument("--ground_truth_column", type=str, dest="ground_truth_column")
    parser.add_argument("--prediction_column", type=str, dest="prediction_column")
    args, _ = parser.parse_known_args()
    return vars(args)


def run():
    """Entry function of model prediction script."""
    args = get_args()
    predictions_data = args["predictions_data"]
    config_str = args["config_str"]
    logger.info(
        f"Running model prediction with data file: {predictions_data}, config_str: {config_str}")
    print(args)
    predictions_data = get_data(predictions_data, y_pred_column_name=args["prediction_column"])
    ground_truth = get_data(args["ground_truth"], y_test_column_name=args["ground_truth_column"])
    config = json.loads(args["config_str"])  # assume config to be dict of different identifiers
    result = pd.DataFrame()
    for identifier in config:
        grader_config = get_config(identifier, config[identifier])
        grader = get_grader(identifier)(grader_config)
        output = grader.compute(predictions_data, ground_truth)
        result = pd.concat([result, output], axis=1)
    result.to_json(args["grader_result"], orient="records", lines=True)


if __name__ == "__main__":
    run()

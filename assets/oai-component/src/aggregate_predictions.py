import json
import logging
import os
from argparse import ArgumentParser

import pandas as pd
from mlflow.utils.proto_json_utils import NumpyEncoder

from aggregator.config.get_config import get_config
from aggregator.get_aggregator import get_aggregator
from data.data_reader import get_data

logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--data", type=str, dest="data", required=True)
    parser.add_argument("--config_str", type=str, dest="config_str", required=True, default=None)
    parser.add_argument("--evaluation_result", type=str, dest="evaluation_result")
    args, _ = parser.parse_known_args()
    return vars(args)


def run():
    """Entry function of model prediction script."""
    args = get_args()
    data = args["data"]
    config_str = args["config_str"]
    evaluation_result = args["evaluation_result"]
    logger.info(
        f"Running model prediction with data file: {data}, config_str: {config_str}")
    data = get_data(data)
    config = json.loads(args["config_str"])  # assume config to be dict of different identifiers
    result = {}
    for identifier in config:
        grader_config = get_config(identifier, config[identifier])
        aggregator = get_aggregator(identifier)(grader_config)
        output = aggregator.aggregate(data)
        result[identifier] = output
    path = args["evaluation_result"]
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.json"), "w") as fp:
        json.dump(result, fp, cls=NumpyEncoder)


if __name__ == "__main__":
    run()

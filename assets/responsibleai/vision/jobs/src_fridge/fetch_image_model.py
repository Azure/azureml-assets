# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import json
import os
import time


import mlflow
import mlflow.pyfunc

from azureml.core import Run

from fastai.learner import load_learner

from raiutils.common.retries import retry_function

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

FRIDGE_MODEL_NAME = 'fridge_model'


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--model_output_path", type=str, help="Path to write model info JSON"
    )
    parser.add_argument(
        "--model_base_name", type=str, help="Name of the registered model"
    )
    parser.add_argument(
        "--model_name_suffix", type=int, help="Set negative to use epoch_secs"
    )
    parser.add_argument(
        "--device", type=int, help=(
            "Device for CPU/GPU supports. Setting this to -1 will leverage "
            "CPU, >=0 will run the model on the associated CUDA device id.")
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


class FetchModel(object):
    def __init__(self):
        pass

    def fetch(self):
        url = ('https://publictestdatasets.blob.core.windows.net/models/' +
               FRIDGE_MODEL_NAME)
        urlretrieve(url, FRIDGE_MODEL_NAME)


def main(args):
    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    _logger.info("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)

    _logger.info("Getting device")
    device = args.device

    _logger.info("Loading parquet input")

    # Load the fridge fastai model
    fetcher = FetchModel()
    action_name = "Model download"
    err_msg = "Failed to download model"
    max_retries = 4
    retry_delay = 60
    retry_function(fetcher.fetch, action_name, err_msg,
                   max_retries=max_retries,
                   retry_delay=retry_delay)
    model = load_learner(FRIDGE_MODEL_NAME)

    if device >= 0:
        model = model.cuda()

    if args.model_name_suffix < 0:
        suffix = int(time.time())
    else:
        suffix = args.model_name_suffix
    registered_name = "{0}_{1}".format(args.model_base_name, suffix)
    _logger.info(f"Registering model as {registered_name}")

    # Saving model with mlflow
    _logger.info("Saving with mlflow")

    mlflow.fastai.log_model(
        model,
        artifact_path=registered_name,
        registered_model_name=registered_name
    )

    _logger.info("Writing JSON")
    dict = {"id": "{0}:1".format(registered_name)}
    output_path = os.path.join(args.model_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(dict, fp=of)


# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import json
import os
import time


import mlflow
import mlflow.pyfunc

import zipfile
from azureml.core import Run

from transformers import AutoModelForSequenceClassification, \
    AutoTokenizer, pipeline

from azureml_model_serializer import PyfuncModel
from raiutils.common.retries import retry_function

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


DBPEDIA_MODEL_NAME = "dbpedia_model"
NUM_LABELS = 9


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
        zipfilename = DBPEDIA_MODEL_NAME + '.zip'
        url = ('https://publictestdatasets.blob.core.windows.net/models/' +
               DBPEDIA_MODEL_NAME + '.zip')
        urlretrieve(url, zipfilename)
        with zipfile.ZipFile(zipfilename, 'r') as unzip:
            unzip.extractall(DBPEDIA_MODEL_NAME)


def retrieve_dbpedia_model():
    fetcher = FetchModel()
    action_name = "Model download"
    err_msg = "Failed to download model"
    max_retries = 4
    retry_delay = 60
    retry_function(fetcher.fetch, action_name, err_msg,
                   max_retries=max_retries,
                   retry_delay=retry_delay)
    model = AutoModelForSequenceClassification.from_pretrained(
        DBPEDIA_MODEL_NAME, num_labels=NUM_LABELS)
    return model


def main(args):
    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    _logger.info("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)

    _logger.info("Getting device")
    device = args.device

    _logger.info("Loading parquet input")

    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = retrieve_dbpedia_model()

    if device >= 0:
        model = model.cuda()

    # build a pipeline object to do predictions
    _logger.info("Buildling model")
    pred = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True
    )

    if args.model_name_suffix < 0:
        suffix = int(time.time())
    else:
        suffix = args.model_name_suffix
    registered_name = "{0}_{1}".format(args.model_base_name, suffix)
    _logger.info(f"Registering model as {registered_name}")

    my_mlflow = PyfuncModel(pred)

    # Saving model with mlflow
    _logger.info("Saving with mlflow")
    mlflow.pyfunc.log_model(
        python_model=my_mlflow,
        registered_model_name=registered_name,
        artifact_path=registered_name,
    )

    _logger.info("Writing JSON")
    model_info = {"id": "{0}:1".format(registered_name)}
    output_path = os.path.join(args.model_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(model_info, fp=of)


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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate running RAITextInsights using responsibleai text docker image."""

import pandas as pd

import datasets
import zipfile
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from raiutils.common.retries import retry_function
from responsibleai_text import RAITextInsights, ModelTask

from spacy.cli import download

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

LABELS = 'labels'
DBPEDIA_MODEL_NAME = "dbpedia_model"
NUM_LABELS = 9


def load_dataset(split):
    dataset = datasets.load_dataset("DeveloperOats/DBPedia_Classes", split=split)
    return pd.DataFrame({"text": dataset["text"], "label": dataset["l1"]})


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


def main():
    """Retrieve model and then run RAITextInsights."""
    NUM_TEST_SAMPLES = 100

    print(download("en_core_web_sm"))

    pd_valid_data = load_dataset("test")

    test_data = pd_valid_data[:NUM_TEST_SAMPLES]

    model = retrieve_dbpedia_model()

    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    device = -1
    if device >= 0:
        model = model.cuda()

    # build a pipeline object to do predictions
    pred = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True
    )

    encoded_classes = ['Agent', 'Device', 'Event', 'Place', 'Species',
                       'SportsSeason', 'TopicalConcept', 'UnitOfWork',
                       'Work']

    rai_insights = RAITextInsights(pred, test_data,
                                   "label",
                                   classes=encoded_classes,
                                   task_type=ModelTask.TEXT_CLASSIFICATION)

    rai_insights.error_analysis.add()
    rai_insights.compute()

    print(rai_insights.error_analysis.get())


# run script
if __name__ == "__main__":
    # run main function
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate running RAIVisionInsights using responsibleai vision docker image."""

import os
import sys
from zipfile import ZipFile
import pandas as pd
from responsibleai_vision import ModelTask, RAIVisionInsights
from responsibleai_vision.common.constants import ImageColumns
from fastai.learner import load_learner
from raiutils.common.retries import retry_function

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


EPOCHS = 10
LEARNING_RATE = 1e-4
IM_SIZE = 300
BATCH_SIZE = 16
FRIDGE_MODEL_NAME = 'fridge_model'
FRIDGE_MODEL_WINDOWS_NAME = 'fridge_model_windows'
WIN = 'win'


def load_fridge_dataset():
    # create data folder if it doesnt exist.
    os.makedirs("data", exist_ok=True)

    # download data
    download_url = ("https://publictestdatasets.blob.core.windows.net/" +
                    "computervision/fridgeObjects.zip")
    data_file = "./data/fridgeObjects.zip"
    urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path="./data")
        print("done")
    # delete zip file
    os.remove(data_file)
    # get all file names into a pandas dataframe with the labels
    data = pd.DataFrame(columns=[ImageColumns.IMAGE.value,
                                 ImageColumns.LABEL.value])
    for folder in os.listdir("./data/fridgeObjects"):
        for file in os.listdir("./data/fridgeObjects/" + folder):
            image_path = "./data/fridgeObjects/" + folder + "/" + file
            data = data.append({ImageColumns.IMAGE.value: image_path,
                                ImageColumns.LABEL.value: folder},
                               ignore_index=True)
    return data


class FetchModel(object):
    def __init__(self):
        pass

    def fetch(self):
        if sys.platform.startswith(WIN):
            model_name = FRIDGE_MODEL_WINDOWS_NAME
        else:
            model_name = FRIDGE_MODEL_NAME
        url = ('https://publictestdatasets.blob.core.windows.net/models/' +
               model_name)
        urlretrieve(url, FRIDGE_MODEL_NAME)


def retrieve_fridge_model():
    fetcher = FetchModel()
    action_name = "Model download"
    err_msg = "Failed to download model"
    max_retries = 4
    retry_delay = 60
    retry_function(fetcher.fetch, action_name, err_msg,
                   max_retries=max_retries,
                   retry_delay=retry_delay)
    model = load_learner(FRIDGE_MODEL_NAME)
    return model


def main():
    """Retrieve model and then run RAIVisionInsights."""
    data = load_fridge_dataset()
    model = retrieve_fridge_model()

    test_data = data
    class_names = data[ImageColumns.LABEL.value].unique()

    rai_insights = RAIVisionInsights(model, test_data,
                                     "label",
                                     task_type=ModelTask.IMAGE_CLASSIFICATION,
                                     classes=class_names)
    rai_insights.compute()


# run script
if __name__ == "__main__":
    # run main function
    main()

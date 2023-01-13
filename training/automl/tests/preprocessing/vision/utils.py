# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision preprocessing utils."""

import logging
import os
import urllib
from zipfile import ZipFile
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


logger = logging.Logger(__name__)


def _download_and_register_image_data(mlclient: MLClient, download_url: str, target_directory: str, asset_name: str):
    """Download image data and register them to ws."""
    data_file = os.path.join(target_directory, (asset_name + ".zip"))
    urllib.request.urlretrieve(download_url, filename=data_file)
    logger.info(f"Downloaded files to {data_file}!!!")
    # extract files
    with ZipFile(data_file, "r") as zip:
        logger.info("extracting files...")
        zip.extractall(path=target_directory)
        logger.info("done")

    my_data = Data(
        name=asset_name,
        path=os.path.join(target_directory, asset_name),
        type=AssetTypes.URI_FOLDER,
    )

    uri_folder_data_asset = mlclient.data.create_or_update(my_data)
    return os.path.join(target_directory, asset_name), uri_folder_data_asset.path

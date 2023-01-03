# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""""""

import os
import urllib
from zipfile import ZipFile
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


def _download_and_register_image_data(mlclient: MLClient, download_url: str, target_directory: str, asset_name: str):
    """ Download image data and register them to ws."""
    data_file = os.path.join(target_directory, (asset_name + ".zip"))
    urllib.request.urlretrieve(download_url, filename=data_file)
    print(f"Downloaded files to {data_file}!!!")
    # extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path=target_directory)
        print("done")
    # Upload data and create a data asset URI folder
    print("Uploading data to blob storage")
    my_data = Data(
        path=os.path.join(target_directory, asset_name),
        type=AssetTypes.URI_FOLDER,
    )
    uri_folder_data_asset = mlclient.data.create_or_update(my_data)
    print(uri_folder_data_asset.path)
    return os.path.join(target_directory, asset_name), uri_folder_data_asset.path

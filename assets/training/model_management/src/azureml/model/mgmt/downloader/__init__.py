# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Downloader init file."""
# flake8: noqa

from .config import ModelSource, MLMODEL, MLFLOW_MODEL, MLFLOW_MODEL_FOLDER
from .downloader import download_model

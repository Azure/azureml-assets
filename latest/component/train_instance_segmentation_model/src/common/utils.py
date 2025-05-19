# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Util functions."""


import json
import os
import shutil
import torch

from azureml.automl.core.shared import log_server, logging_utilities
from azureml.automl.core.shared.constants import MLTableDataLabel, MLTableLiterals
from azureml.automl.dnn.vision.common import utils
from azureml.automl.dnn.vision.common.exceptions import AutoMLVisionValidationException
from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.common.constants import SettingsLiterals
from azureml.core import Run

from common.settings import CommonSettings


logger = get_logger('azureml.automl.dnn.vision.asset_registry')


def create_component_telemetry_wrapper(task_type):
    """Create a decorator for components that tracks telemetry."""

    def component_telemetry_wrapper(func):
        def wrapper(*args, **kwargs):
            # Initialize logging
            settings = {SettingsLiterals.TASK_TYPE: task_type}
            utils._top_initialization(settings)
            utils._set_logging_parameters(task_type, settings)

            # Set logging dimension so we can distinguish all logs coming from component.
            log_server.update_custom_dimensions({'training_component': True})

            try:
                logger.info('Training component started')
                with logging_utilities.log_activity(logger, activity_name='TrainingComponent'):
                    result = func(*args, **kwargs)
                logger.info('Training component succeeded')
                return result
            except Exception as e:
                logger.warning('Training component failed')
                logging_utilities.log_traceback(e, logger)
                raise
            finally:
                logger.info('Training component completed')
        return wrapper
    return component_telemetry_wrapper


def create_mltable_json(settings: CommonSettings) -> str:
    """Create MLTable in JSON."""
    mltable_data_dict = {
        MLTableDataLabel.TrainData.value: {
            MLTableLiterals.MLTABLE_RESOLVEDURI: settings.training_data
        }
    }

    if settings.validation_data:
        mltable_data_dict[MLTableDataLabel.ValidData.value] = {
            MLTableLiterals.MLTABLE_RESOLVEDURI: settings.validation_data
        }

    return json.dumps(mltable_data_dict)


def get_local_rank() -> int:
    """Get local rank."""
    return int(os.environ["LOCAL_RANK"])


def validate_running_on_gpu_compute() -> None:
    """Check if GPU compute is available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise AutoMLVisionValidationException(
            "This component requires compute that contains one or more GPU.")


def download_models(run: Run, mlflow_output: str, pytorch_output: str):
    """Download models."""
    TMP_OUTPUT = '/tmp/outputs'
    TMP_MLFLOW = TMP_OUTPUT + '/mlflow-model'

    run.download_files(
        prefix='outputs', output_directory=TMP_OUTPUT, append_prefix=False)

    # Copy the mlflow model
    try:
        shutil.copytree(TMP_MLFLOW, mlflow_output, dirs_exist_ok=True)
    except Exception as e:
        logger.error('Error in uploading mlflow model: {}'.format(e))

    shutil.rmtree(TMP_MLFLOW, ignore_errors=True)

    # Copy the pytorch model
    try:
        shutil.copytree(TMP_OUTPUT, pytorch_output, dirs_exist_ok=True)
    except Exception as e:
        logger.error('Error in uploading pytorch model: {}'.format(e))

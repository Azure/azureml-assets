# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
File containing function for score.
"""

import os
import json
import argparse

import numpy as np

from azureml.train.finetune.core.constants.constants import SaveFileConstants
from azureml.train.finetune.core.drivers.deployment import Deployment
from azureml.train.finetune.core.utils.logging_utils import get_logger_app
from azureml.train.finetune.core.utils.decorators import swallow_all_exceptions
from azureml.train.finetune.core.utils.error_handling.exceptions import ResourceException
from azureml.train.finetune.core.utils.error_handling.error_definitions import DeploymentFailed
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

logger = get_logger_app()

DEPLOY_OBJ = None


class _JSONEncoder(json.JSONEncoder):
    """
    custom `JSONEncoder` to make sure float and int64 ar converted
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(_JSONEncoder, self).default(obj)


def encode_json(content):
    """
    encodes json with custom `JSONEncoder`
    """
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        cls=_JSONEncoder,
        separators=(",", ":"),
    )


def decode_json(content):
    """
    decode the json content
    """
    return json.loads(content)


@swallow_all_exceptions(logger)
def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global DEPLOY_OBJ

    env_var = json.loads(os.environ[SaveFileConstants.DeploymentSaveKey])
    env_var["model_path"] = os.environ.get("AZUREML_MODEL_DIR", None)

    try:
        model_path = env_var["model_path"]
        parent_dir_name = env_var["parent_dir_name"]
        model_parent_dir_name = os.path.basename(model_path)
        if model_parent_dir_name != parent_dir_name:
            model_path = os.path.join(model_path, parent_dir_name)
        env_var["model_path"] = model_path
        logger.info(f"Model path - {model_path}")
        # model directory contents
        for dirpath, _, filenames in os.walk(model_path):
            for filename in filenames:
                logger.info(os.path.join(dirpath, filename))
        args = argparse.Namespace(**env_var)
        logger.info(args)
        DEPLOY_OBJ = Deployment(args)
        # initialize tokenizer and model for prediction
        DEPLOY_OBJ.prepare_prediction_service()
    except Exception as e:
        raise ResourceException._with_error(
            AzureMLError.create(DeploymentFailed, error=e)
        )


@swallow_all_exceptions(logger)
def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    `raw_data` is the raw request body data.
    """
    try:
        data = decode_json(raw_data)
        # pop inputs for pipeline
        inputs = data.pop("inputs", data)
        predictions = DEPLOY_OBJ.predict(inputs)
    except Exception as e:
        # we should never terminate score script by raising exception in run()
        # as it need to continously serve online requests
        # traceback.print_exc()
        predictions = {"msg": "failed", "error": str(e)}
        logger.error("Exception: \n", exc_info=True)
    return encode_json(predictions)

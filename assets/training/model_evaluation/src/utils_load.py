# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model Evaluation model load utilities."""

import azureml.evaluate.mlflow as aml_mlflow
import constants
import torch
from logging_utilities import get_logger
from mlflow.models import Model

logger = get_logger(name=__name__)


def load_model(model_uri, device, task):
    """Load model from given details.

    Args:
        model_uri (_type_): _description_
        device (_type_): _description_
        task (_type_): _description_

    Returns:
        model: _description_
    """
    curr_model = Model.load(model_uri).flavors
    if "hftransformers" in curr_model:
        curr_model = curr_model.get("hftransformers")
    elif "hftransformersv2" in curr_model:
        curr_model = curr_model.get("hftransformersv2")
    else:
        curr_model = {}
    aml_args = {
        "model_hf_load_kwargs": curr_model.get("model_hf_load_kwargs", {})
    }
    if device == constants.DEVICE.AUTO and torch.cuda.is_available():
        aml_args["model_hf_load_kwargs"]["device_map"] = constants.DEVICE.AUTO
    elif device == constants.DEVICE.GPU and torch.cuda.is_available():
        aml_args["model_hf_load_kwargs"]["device_map"] = torch.cuda.current_device()
    elif isinstance(device, int) and device >= 0:
        aml_args["model_hf_load_kwargs"]["device_map"] = device
    else:
        if device == constants.DEVICE.GPU:
            logger.warning("Device_map set as GPU, but the compute doesn't have a GPU.")
        logger.info("Loading model on CPU with f32 dtype")
        aml_args["model_hf_load_kwargs"]["device_map"] = constants.DEVICE.CPU
        aml_args["model_hf_load_kwargs"]["torch_dtype"] = torch.float32

    try:
        logger.info(f"aml args: {aml_args}")
        model = aml_mlflow.aml.load_model(model_uri=model_uri,
                                          model_type=constants.MLFLOW_MODEL_TYPE_MAP[task], **aml_args)
    except Exception:
        logger.info("Reloading model with device_map NA")
        aml_args["model_hf_load_kwargs"]["device_map"] = "eval_na"
        logger.info(f"aml args: {aml_args}")
        model = aml_mlflow.aml.load_model(model_uri=model_uri,
                                          model_type=constants.MLFLOW_MODEL_TYPE_MAP[task], **aml_args)

    return model

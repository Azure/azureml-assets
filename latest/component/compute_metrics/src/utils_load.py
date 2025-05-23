# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model Evaluation model load utilities."""

import os
import azureml.evaluate.mlflow as aml_mlflow
import mlflow
import constants
import torch
from logging_utilities import get_logger
from mlflow.models import Model
from constants import MODEL_FLAVOR

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
    model_flavor = ""

    if "hftransformers" in curr_model:
        curr_model = curr_model.get("hftransformers")
        model_flavor = MODEL_FLAVOR.HFTRANSFORMERS
    elif "hftransformersv2" in curr_model:
        curr_model = curr_model.get("hftransformersv2")
        model_flavor = MODEL_FLAVOR.HFTRANSFORMERSV2
    elif "transformers" in curr_model:
        model_flavor = MODEL_FLAVOR.TRANSFORMERS
    else:
        curr_model = {}
    aml_args = {
        "model_hf_load_kwargs": curr_model.get("model_hf_load_kwargs", {})
    }
    # Todo: Remove this once we have a fix for the issue
    if model_flavor == MODEL_FLAVOR.TRANSFORMERS and constants.MLFLOW_MODEL_TYPE_MAP[task] == "summarization":
        logger.info("setting OS for text-summarization task")
        os.environ["MLFLOW_HUGGINGFACE_USE_DEVICE_MAP"] = "False"

    if device == constants.DEVICE.AUTO and torch.cuda.is_available():
        aml_args["model_hf_load_kwargs"]["device_map"] = constants.DEVICE.AUTO
    elif device == constants.DEVICE.GPU and torch.cuda.is_available():
        aml_args["model_hf_load_kwargs"]["device_map"] = torch.cuda.current_device()
        os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"] = str(torch.cuda.current_device())
    elif isinstance(device, int) and device >= 0:
        aml_args["model_hf_load_kwargs"]["device_map"] = device
        os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"] = str(device)
    else:
        if device == constants.DEVICE.GPU:
            logger.warning("Device_map set as GPU, but the compute doesn't have a GPU.")
        logger.info("Loading model on CPU with f32 dtype")
        aml_args["model_hf_load_kwargs"]["device_map"] = constants.DEVICE.CPU
        # Todo: Add equivalent for tramformers flavor as well
        # Check what case fails here
        aml_args["model_hf_load_kwargs"]["torch_dtype"] = torch.float32
        os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"] = str(-1)
    try:
        logger.info(f"aml args: {aml_args}")
        if model_flavor != MODEL_FLAVOR.TRANSFORMERS:
            logger.info("Loading model in hftransformers flavor")
            model = aml_mlflow.aml.load_model(model_uri=model_uri,
                                              model_type=constants.MLFLOW_MODEL_TYPE_MAP[task], **aml_args)
        else:
            logger.info("Loading model in mlflow transformers flavor")
            model = mlflow.pyfunc.load_model(model_uri)
    except Exception:
        logger.info("Reloading model with device_map NA")
        if model_flavor != MODEL_FLAVOR.TRANSFORMERS:
            logger.info("Loading model in hftransformers flavor")
            aml_args["model_hf_load_kwargs"]["device_map"] = "eval_na"
            logger.info(f"aml args: {aml_args}")
            model = aml_mlflow.aml.load_model(model_uri=model_uri,
                                              model_type=constants.MLFLOW_MODEL_TYPE_MAP[task], **aml_args)
        else:
            os.environ["MLFLOW_HUGGINGFACE_USE_DEVICE_MAP"] = "False"
            logger.info("Loading model in mlflow transformers flavor")
            model = mlflow.pyfunc.load_model(model_uri)

    return model, model_flavor

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model Registration module."""
import time
import argparse
from argparse import Namespace
from typing import Dict, Optional

import json
import re
import requests
from random import randint
from pathlib import Path
from azureml.core.model import Model

from azureml.core import Workspace
from azureml.core.run import Run, _OfflineRun
from azureml._model_management._util import get_requests_session
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml._restclient.clientbase import ClientBase

logger = get_logger_app("azureml.acft.contrib.hf.scripts.components.scripts.register_model.register_model")


COMPONENT_NAME = "ACFT-Register_Model"
SUPPORTED_MODEL_ASSET_TYPES = [Model.Framework.CUSTOM, "PRESETS"]
# omitting underscores which is supported in model name for consistency
VALID_MODEL_NAME_PATTERN = r"^[a-zA-Z0-9-]+$"
NEGATIVE_MODEL_NAME_PATTERN = r"[^a-zA-Z0-9-]"
REGISTRATION_DETAILS_JSON_FILE = "model_registration_details.json"
DEFAULT_MODEL_NAME = "default_model_name"
RETRY_NUM_FOR_409 = 10

def str2bool(arg):
    """Convert string to bool."""
    arg = arg.lower()
    if arg in ["true", '1']:
        return True
    elif arg in ["false", '0']:
        return False
    else:
        raise ValueError(f"Invalid argument {arg} to while converting string to boolean")


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_path", type=str, help="Directory containing model files")
    parser.add_argument(
        "--convert_to_safetensors",
        type=str2bool,
        default="false",
        choices=[True, False],
        help="convert pytorch model to safetensors format"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="Finetuning task name",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=Model.Framework.CUSTOM,
        choices=SUPPORTED_MODEL_ASSET_TYPES,
        help="Type of model you want to register",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name to use for the registered model. If it already exists, the version will be auto incremented.",
    )
    parser.add_argument(
        "--finetune_args_path",
        type=str,
        help="JSON file that contains the finetune information",
        default=None,
    )
    parser.add_argument(
        "--model_version",
        type=str,
        help="Model version in workspace/registry. If model with same version exists,version will be auto incremented",
        default=None,
    )
    parser.add_argument(
        "--registration_details_folder",
        type=Path,
        help="A folder which contains a JSON file into which model registration details will be written",
    )
    args = parser.parse_args()
    logger.info(f"Args received {args}")
    return args


def get_workspace_details() -> Workspace:
    """Fetch the workspace details from run context."""
    run = Run.get_context()
    if isinstance(run, _OfflineRun):
        return Workspace.from_config()
    return run.experiment.workspace


def is_model_registered_already(request_uri, model_properties, params, headers):
    current_run_id = model_properties["runId"]
    model_name = model_properties["name"]
    model_version = model_properties["version"]
    url = request_uri + "/" + model_name + ":" + model_version
    try:
        resp = ClientBase._execute_func(get_requests_session().get, url, params=params, headers=headers)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        # log any HTTP errors and return False
        logger.error('Received bad response from GET Model request:\n'
                        'Response Code: {}\n'
                        'Headers: {}\n'
                        'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        return False
    # check if model is registered with the same runId as current.
    # if it is then model registration succeeded in past so return True.
    # if not then some other model is registered with same name and version so return False.
    return  "runId" in resp.json() and resp.json()["runId"] == current_run_id


def submit_rest_request(request_type, request_uri, request_body, params=None, headers=None, use_auto_version=False):
    # TODO: This relies on the v1 SDK. Handling will need to be adapted to shift to v2

    retry_count = RETRY_NUM_FOR_409 if use_auto_version else 1
    i = 0
    while True:
        logger.info(f"Attempt number {i + 1} for model registration.")
        try:
            resp = ClientBase._execute_func(request_type, request_uri, params=params, headers=headers, json=request_body)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError:
            # check if model has been registered already
            if resp.status_code == 409:
                logger.warning('This model id (name:version combination) already exists.')
                logger.info('Checking if this model has already been registered by current RunId previous attempts/retries.')
                if not use_auto_version and is_model_registered_already(request_uri, request_body, params, headers):
                    logger.info('Good news! This model has already been registered by current RunId.')
                    return resp
                i += 1
                logger.error('This model id has already been used for a different model registration.')
                if i == retry_count:
                    logger.error('Exceeded all retry attempts.')
                else:
                    logger.error(f'Attempting to retry again with autoVersion set to True.')
                    time.sleep(randint(1000, 5000)/1000) # sleep randomly from 1 second to 5 seconds
                    continue

            raise Exception('Received bad response from POST Model request:\n'
                            'Response Code: {}\n'
                            'Headers: {}\n'
                            'Content: {}'.format(resp.status_code, resp.headers, resp.content))


def get_modelregistry_url(workspace):
    from azureml._restclient.assets_client import AssetsClient
    assets_client = AssetsClient(workspace.service_context)
    modelregistry_url = assets_client.get_cluster_url()
    uri = '/modelmanagement/{}/subscriptions/{}/resourceGroups/{}/providers/' \
          'Microsoft.MachineLearningServices/workspaces/{}/models'.format("v1.0",
                                                                   workspace.subscription_id,
                                                                   workspace.resource_group,
                                                                   workspace.name)
    return modelregistry_url + uri


def get_pipeline_run_id():
    """Fetch pipeline runId at top level."""
    run = Run.get_context()
    try:
        run_details = run.get_details()
        
        # get pipeline run id
        run_id = run_details['runId']
        top_level_run = run
        while top_level_run.parent:
            top_level_run = top_level_run.parent
        return top_level_run.id
    except Exception:
        logger.warning(f"cannot fetch top level run, returning none.")
        return None


def get_model_path_in_HOBO_storage():
    run = Run.get_context()
    try:
        run_details = run.get_details()
        model_path_in_storage = run_details['runDefinition']['outputData']['output_model']['outputLocation']['uri']['path']
        return model_path_in_storage
    except Exception:
        logger.warning(f"cannot fetch model output path, returning none.")
        return None


def is_model_available(ml_client, model_name, model_version):
    """Return true if model is available else false."""
    is_available = True
    try:
        ml_client.models.get(name=model_name, version=model_version)
    except Exception as e:
        logger.warning(f"Model with name - {model_name} and version - {model_version} is not available. Error: {e}")
        is_available = False
    return is_available


def get_model_name(finetune_args_path: str) -> Optional[str]:
    """Construct the model name from the base model."""
    import uuid
    with open(finetune_args_path, 'r', encoding="utf-8") as rptr:
        finetune_args_dict = json.load(rptr)

    try:
        base_model_name = finetune_args_dict.get("model_asset_id").split("/")[-3]
    except Exception:
        base_model_name = DEFAULT_MODEL_NAME
    logger.info(f"Base model name: {base_model_name}")

    new_model_name = base_model_name + "-ft-" + str(uuid.uuid4())
    logger.info(f"Updated model name: {new_model_name}")

    return new_model_name


def convert_lora_weights_to_safetensors(model_path: str):
    """Read the bin files and convert them to safe tensors."""
    import os
    import torch
    from azureml.acft.contrib.hf.nlp.utils.io_utils import find_files_with_inc_excl_pattern
    from safetensors.torch import save_file

    bin_files = find_files_with_inc_excl_pattern(model_path, include_pat=".bin$")
    logger.info(f"Following bin files are identified: {bin_files}")
    for bin_file in bin_files:
        bin_file_sd = torch.load(bin_file, map_location=torch.device("cpu"))
        safe_tensor_file = bin_file.replace(".bin", ".safetensors")
        save_file(bin_file_sd, safe_tensor_file)
        logger.info(f"Created {safe_tensor_file}")
        os.remove(bin_file)
        logger.info(f"Deleted {bin_file}")


def copy_model_to_output(model_path: str, output_dir: str):
    """Copy the model from model path to output dir."""
    import shutil
    logger.info("Started copying the model weights to output directory")
    print(f"Copying the model from local path {model_path} to output {output_dir}")
    shutil.copytree(model_path, output_dir, dirs_exist_ok=True)
    logger.info("Completed copying the weights")
    print(f"Done copied the model from local path {model_path} to output {output_dir}")


def get_properties(finetune_args_path: str) -> Dict[str, str]:
    """Fetch the appropriate properties regarding the base model."""
    properties = {}
    with open(finetune_args_path, 'r', encoding="utf-8") as rptr:
        finetune_args_dict = json.load(rptr)

    # read from finetune config
    property_key_to_finetune_args_key_map = {
        "baseModelId": "model_asset_id",
    }
    for property_key, finetune_args_key in property_key_to_finetune_args_key_map.items():
        properties[property_key] = finetune_args_dict.get(finetune_args_key, None)
        if "baseModelId" == property_key:
            properties[property_key] = "/".join(properties[property_key].split('/')[:-2])

    # fixed properties
    additional_properties = {
        "baseModelWeightsVersion": 1.0,
    }
    properties.update(additional_properties)
    logger.info(f"Adding the following properties to the registered model: {properties}")

    return properties


def register_model(args: Namespace):
    """Run main function for sdkv1."""
    model_name = args.model_name
    model_type = args.model_type
    registration_details_folder = args.registration_details_folder
    tags, properties, model_description = {}, {}, ""

    # set properties
    properties = get_properties(args.finetune_args_path)

    # create workspace details
    ws = get_workspace_details()

    if not re.match(VALID_MODEL_NAME_PATTERN, model_name):
        # update model name to one supported for registration
        logger.info(f"Updating model name to match pattern `{VALID_MODEL_NAME_PATTERN}`")
        model_name = re.sub(NEGATIVE_MODEL_NAME_PATTERN, "-", model_name)
        logger.info(f"Updated model_name = {model_name}")

    st = time.time()
    print(f"registering the model using model from {registration_details_folder}")
    # calling into model registry by using REST API
    request_url = get_modelregistry_url(ws)
    model_path_in_HOBO = get_model_path_in_HOBO_storage()
    top_level_run_id = get_pipeline_run_id()
    request_body = {
        "name": model_name,
        "description": model_description,
        "url": model_path_in_HOBO,
        "runId": top_level_run_id,
        "mimeType": "application/x-python",
        "properties": properties,
        "kvTags": tags,
        "modelFormat": model_type
    }
    print(f"request uri is {request_url}, request body is: {request_body}")
    request_params = {'autoVersion': 'true'}
    request_headers = {'Content-Type': 'application/json'}
    request_headers.update(ws._auth_object.get_authentication_header())
    logger.info('Starting register model request')
    resp = submit_rest_request(get_requests_session().post, request_url, request_body,
                               request_params, request_headers, use_auto_version=True)
    logger.info('Done register model request')
    resp_json = resp.json()
    print("Response json:", resp_json)
    if 'name' in resp_json and 'version' in resp_json:
        logger.info(f"Registered model {resp_json['name']} with version {resp_json['version']}")

    # model = Model.register(
    #     workspace=ws,
    #     model_path=registration_details_folder,  # where the model was copied to in output
    #     model_name=model_name,
    #     model_framework=model_type,
    #     description=model_description,
    #     tags=tags,
    #     properties=properties
    # )
    time_to_register = time.time() - st
    logger.info(f"Time to register: {time_to_register} seconds")

    # register the model in workspace or registry
    # logger.info(f"Registering model {model.name} with version {model.version}.")
    # logger.info(f"Model registered. AssetID : {model.id}")
    # Registered model information
    # model_info = {
    #     "id": model.id,
    #     "name": model.name,
    #     "version": model.version,
    #     "type": model.model_framework,
    #     "properties": model.properties,
    #     "tags": model.tags,
    #     "description": model.description,
    # }
    json_object = json.dumps(resp_json, indent=4)

    registration_file = registration_details_folder / REGISTRATION_DETAILS_JSON_FILE

    with open(registration_file, "w+") as outfile:
        outfile.write(json_object)
    logger.info("Saved model registration details in output json file.")


# run script
if __name__ == "__main__":
    args = parse_args()

    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    # convert to safe tensors
    if args.convert_to_safetensors:
        convert_lora_weights_to_safetensors(args.model_path)

    # update model name
    if args.model_name is None:
        args.model_name = get_model_name(args.finetune_args_path)

    # copy to output dir
    copy_model_to_output(args.model_path, args.registration_details_folder)

    # register model
    register_model(args)

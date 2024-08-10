# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Scoring script to infer DeepSpeed MII deployed model."""

import json
import torch
import mii
from pathlib import Path


model = None
PARENT_DIR = Path().parent.resolve()
MODEL_INFO_FILE = "model_details.json"


def get_mii_configs():
    """Return MII config."""
    model_info_path = PARENT_DIR / MODEL_INFO_FILE
    with open(model_info_path) as f:
        model_info = json.load(f)

    model_name = model_info["model_name"]
    task_name = model_info["task_name"]

    return {
        'ds_config': None,
        'ds_optimize': True,
        'ds_zero': False,
        'mii_configs': {
            'checkpoint_dict': None,
            'deploy_rank': [0],
            'dtype': torch.float32,
            'enable_cuda_graph': False,
            'hf_auth_token': None,
            'load_with_sys_mem': False,
            'port_number': 50050,
            'profile_model_time': False,
            'replace_with_kernel_inject': True,
            'skip_model_check': False,
            'tensor_parallel': 1,
            'torch_dist_port': 29500
        },
        'model_path': '',
        'model_name': model_name,
        'task_name': task_name,
    }


def init():
    """Init MII server."""
    print(f"init(): configs=> \n{configs}")
    model_path = mii.utils.full_model_path(configs.get(mii.constants.MODEL_PATH_KEY, ""))

    model_name = configs[mii.constants.MODEL_NAME_KEY]
    task = configs[mii.constants.TASK_NAME_KEY]

    assert model_name is not None, "The model name should be set before calling init"
    assert task is not None, "The task name should be set before calling init"

    mii.MIIServer(task,
                  model_name,
                  model_path,
                  ds_optimize=configs[mii.constants.ENABLE_DEEPSPEED_KEY],
                  ds_zero=configs[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY],
                  ds_config=configs[mii.constants.DEEPSPEED_CONFIG_KEY],
                  mii_configs=configs[mii.constants.MII_CONFIGS_KEY])

    global model
    model = None

    # In AML deployments both the GRPC client and server are used in the same process
    if mii.utils.is_aml():
        model = mii.MIIClient(task, mii_configs=configs[mii.constants.MII_CONFIGS_KEY])
        print(f"Model:\n\n {model}\n\n")


def run(request):
    """Invoke MII query and return response."""
    global model
    assert model is not None, "grpc client has not been setup when this model was created"
    request_dict = json.loads(request)

    print(f"request_dict {request_dict}\n\n")
    query_dict = mii.utils.extract_query_dict(configs[mii.constants.TASK_NAME_KEY],
                                              request_dict)
    print(f"query_dict {query_dict}\n\n")

    response = model.query(query_dict, **request_dict)
    time_taken = response.time_taken

    print(f"type(response): {type(response)}")
    if hasattr(response, "response"):
        print(f"type(response.response) => {type(response.response)}")
    if not isinstance(response.response, str):
        response = [r for r in response.response]
    if not isinstance(response.response, str):
        response = [r for r in response.response]
    result_dict = {'responses': response.response, 'time': time_taken}
    print(result_dict)
    return json.dumps(result_dict)


configs = get_mii_configs()
print(configs)

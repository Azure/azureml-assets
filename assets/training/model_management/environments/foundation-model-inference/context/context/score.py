import os
import json
import torch
import mii
from mii.config import LoadBalancerConfig, ReplicaConfig
import time
import logging

model = None
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
format_str = "%(asctime)s [%(module)s] %(funcName)s %(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info(f"Environment Variables:")
for name, value in os.environ.items():
    logger.info(f"{name}: {value}")

LOAD_BALANCING_PORT = 50050
MAX_TOKENS = 1024
TORCH_DIST_PORT = 29501
REPLICA_NUM = int(os.getenv("WORKER_COUNT", 1))
DEVICE_COUNT = torch.cuda.device_count()
TENSOR_PARALLEL = int(DEVICE_COUNT / REPLICA_NUM)
MODEL_NAME = os.getenv("AZUREML_MODEL_DIR")
MODEL_PATH = "mlflow_model_folder/data/model"


def init():
    model_path = mii.utils.full_model_path(configs[mii.constants.MODEL_PATH_KEY])
    deployment_name = configs[mii.constants.DEPLOYMENT_NAME_KEY]
    model_name = configs[mii.constants.MODEL_NAME_KEY]
    task_name = configs[mii.constants.TASK_NAME_KEY]

    assert model_name is not None, "The model name should be set before calling init"
    assert task_name is not None, "The task name should be set before calling init"

    try:
        start_server = True
        if int(os.getpid()) % configs.get("mii_configs").get("replica_num") != 0:
            start_server = False
            logger.info("Skip MII server setup for this process")

        if start_server:
            logger.info("Start server setup")
            mii.MIIServer(deployment_name,
                        task_name,
                        model_name,
                        model_path,
                        ds_optimize=configs[mii.constants.ENABLE_DEEPSPEED_KEY],
                        ds_zero=configs[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY],
                        ds_config=configs[mii.constants.DEEPSPEED_CONFIG_KEY],
                        mii_configs=configs[mii.constants.MII_CONFIGS_KEY],
                        lb_config=configs.get(mii.constants.LOAD_BALANCER_CONFIG_KEY,
                                                None))
            logger.info("Completed server setup")
        
    except Exception as e:
        logger.warning(f"MIIServer setup failed. Error {e}")
    
    logger.info("Start client setup")

    global model
    model = None

    # In AML deployments both the GRPC client and server are used in the same process
    try:
        model = mii.MIIClient(task_name, "localhost", configs.get("mii_configs").get("port_number"))
    except Exception as e:
        logger.warning(f"MIIClient setup failed. Error {e}")
    logger.info("Completed client setup")


def run(data):
    global model
    assert model is not None, "grpc client has not been setup when this model was created"
    
    try:
        request_dict = json.loads(data)

        print(f"request_dict {request_dict}\n\n")
        query_dict = mii.utils.extract_query_dict(configs[mii.constants.TASK_NAME_KEY], request_dict)
        print(f"query_dict {query_dict}\n\n")

        response = model.query(query_dict, **request_dict)
        time_taken = response.time_taken

        result_dict = {'responses': f"responses: {response.response}", 'time': time_taken}

        return json.dumps(result_dict)

    except Exception as e:
        return json.dumps({"error": str(e)})


def allocate_processes(hostfile_path):
    from mii.deployment import _allocate_processes
    if hostfile_path is None:
        import tempfile
        hostfile_path = tempfile.NamedTemporaryFile(delete=False).name
        logger.info(f"hostfile_path: {hostfile_path}")
        num_gpu = DEVICE_COUNT
        with open(hostfile_path, "w") as f:
            f.write(f"localhost slots={num_gpu}")
    return _allocate_processes(hostfile_path, TENSOR_PARALLEL, REPLICA_NUM)


def generate_load_balancer_config():
    replica_pool = allocate_processes(hostfile_path=None)    
    replica_configs = [
        ReplicaConfig(
            hostname=hostname,
            tensor_parallel_ports=list(range(LOAD_BALANCING_PORT+i*TENSOR_PARALLEL+1, LOAD_BALANCING_PORT+i*TENSOR_PARALLEL+1+TENSOR_PARALLEL)),
            torch_dist_port=i+TORCH_DIST_PORT,
            gpu_indices=gpu_indices
        ) 
        for i, (hostname, gpu_indices) in enumerate(replica_pool)
    ]
    load_balancer_config = LoadBalancerConfig(port=LOAD_BALANCING_PORT, replica_configs=replica_configs)
    return load_balancer_config


load_balancer_config = generate_load_balancer_config()

configs = {
    'deployment_name': 'llama-deployment',
    'ds_config': None,
    'ds_optimize': True,
    'ds_zero': False,
    'load_balancer_config': load_balancer_config,
    'mii_configs': {
        'checkpoint_dict': None,
        'deploy_rank': load_balancer_config.replica_configs[0].gpu_indices,
        'dtype': torch.float16,
        'enable_cuda_graph': False,
        'enable_restful_api': False,
        'hf_auth_token': None,
        'load_with_sys_mem': True,
        'max_tokens': MAX_TOKENS,
        'meta_tensor': False,
        'port_number': LOAD_BALANCING_PORT,
        'profile_model_time': False,
        'replace_with_kernel_inject': True,
        'replica_num': REPLICA_NUM,
        'skip_model_check': True,
        'tensor_parallel': TENSOR_PARALLEL,
        'torch_dist_port': TORCH_DIST_PORT,
        'trust_remote_code': False
    },
    'model_name': MODEL_NAME,
    'model_path': MODEL_PATH,
    'task_name': 'text-generation'
}

logger.info(f"MII configs: {configs}")

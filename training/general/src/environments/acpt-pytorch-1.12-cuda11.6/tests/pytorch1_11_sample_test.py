# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the pytorch 1.12 environment."""
import os
import time
from pathlib import Path
from azure.ai.ml import command, Output, MLClient, PyTorchDistribution
from azure.ai.ml.entities import Environment, BuildContext, JobResourceConfiguration
from azure.identity import AzureCliCredential
import subprocess

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 60)
STD_LOG = Path("artifacts/user_logs/std_log.txt")


def test_pytorch_1_12():
    """Tests a sample job using pytorch 1.12 as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "acpt-pytorch-1_12-cuda11_6"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="Pytorch 1.12 environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="pip install -r requirements.txt" \
                " && python pretrain_glue.py --tensorboard_log_dir \"/outputs/runs/\"" \
                " --deepspeed ds_config.json --num_train_epochs 5 --output_dir outputs --disable_tqdm 1" \
                " --local_rank $RANK --evaluation_strategy \"epoch\" --logging_strategy \"epoch\"" \
                " --per_device_train_batch_size 93 --gradient_accumulation_steps 1" \
                " --per_device_eval_batch_size 93 --learning_rate 3e-05 --adam_beta1 0.8 --adam_beta2 0.999" \
                " --weight_decay 3e-07 --warmup_steps 500 --fp16 --logging_steps 1000" \
                " --model_checkpoint \"bert-large-uncased\"",
        outputs={
            "output": Output(
                type="uri_folder",
                mode="rw_mount",
                path="azureml://datastores/workspaceblobstore/paths/outputs"
            )
        },
        environment=f"{env_name}@latest",
        compute=os.environ.get("gpu_v100_cluster"),
        display_name="bert-pretrain-GLUE",
        description="Pretrain the BERT model on the GLUE dataset.",
        experiment_name="pytorch112_Cuda116_Experiment",
        distribution=PyTorchDistribution(process_count_per_instance=1),
        resources=JobResourceConfiguration(instance_count=2, shm_size='3100m'),
    )

    returned_job = ml_client.create_or_update(job)
    assert returned_job is not None

    # Poll until final status is reached or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        current_status = ml_client.jobs.get(returned_job.name).status
        if current_status in ["Completed", "Failed"]:
            break
        time.sleep(30)  # sleep 30 seconds

    bashCommand = "ls"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    print(error)

    if current_status == "Failed":
        ml_client.jobs.download(returned_job.name)
        if STD_LOG.exists():
            print(f"*** BEGIN {STD_LOG} ***")
            with open(STD_LOG, "r") as f:
                print(f.read(), end="")
            print(f"*** END {STD_LOG} ***")
        else:
            ml_client.jobs.stream(returned_job.name)

    assert current_status == "Completed"

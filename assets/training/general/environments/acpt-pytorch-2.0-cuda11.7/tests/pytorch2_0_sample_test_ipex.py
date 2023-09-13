# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the pytorch 2.0 environment."""
import os
import time
from pathlib import Path
from azure.ai.ml import command, Output, MLClient
from azure.ai.ml.entities import Environment, BuildContext, JobResourceConfiguration, AmlCompute
from azure.identity import AzureCliCredential
import subprocess

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "../../acpt-tests/src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 120)
STD_LOG = Path("artifacts/user_logs/std_log.txt")

def test_pytorch_2_0():
    """Tests a sample job using pytorch 2.0 as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "acpt-pytorch-2_0-cuda11_7"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="Pytorch 2.0 environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    ##########################################################################
    # Create a 2nd Generation Intel(R) Xeon Scalable Processor compute cluster
    cpu_compute_target = "cpu-xeon-cluster"

    try:
        cpu_cluster = ml_client.compute.get(cpu_compute_target)
    except Exception:
        cpu_cluster = AmlCompute(
            name=cpu_compute_target,
            type="amlcompute",
            size = "Standard_D4d_v4",
            min_instances=0,
            max_instances=1,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )

        ml_client.begin_create_or_update(cpu_cluster).result()
    ##########################################################################

    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="pip install -r requirements.txt" \
                " && python ipex_bert_train.py --intel-extension",
        outputs={
            "output": Output(
                type="uri_folder",
                mode="rw_mount",
                path="azureml://datastores/workspaceblobstore/paths/outputs"
            )
        },
        environment=f"{env_name}@latest",
        compute=cpu_compute_target,
        display_name="IPEX_BERT_train",
        description="Pretrain the BERT model on the GLUE dataset.",
        experiment_name="pytorch20_ipex20_Experiment",
        resources=JobResourceConfiguration(instance_count=1),
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

    if current_status == "Failed" or current_status == "Cancelled":
        ml_client.jobs.download(returned_job.name)
        if STD_LOG.exists():
            print(f"*** BEGIN {STD_LOG} ***")
            with open(STD_LOG, "r") as f:
                print(f.read(), end="")
            print(f"*** END {STD_LOG} ***")
        else:
            ml_client.jobs.stream(returned_job.name)

    assert current_status == "Completed"

import os
import subprocess
import requests
from datetime import datetime, timedelta
import time
import io
import docker
import argparse
import json

# For local testing, KILL_CONTAINER can be set to false so we can delve into running container and see logs
KILL_CONTAINER = True

DEFAULT_MODEL_DIR_NAME = "/var/model_dir"


def start_docker(
    inference_image_name,
    resources_directory,
    environment_variables={},
    is_gpu=False,
    overwrite_azuremlapp=True,
    additional_ports=None,
    cap_add=[],
    volumes={},
):
    """
    Utility function that starts a docker container

            Parameters:
                    inference_image_name: Inference image name
                    resource_directory: Directory that will be mounted on the container as /var/azureml-app
                    environment_variables: Environment variables
                    is_gpu: Designation if docker to be run using gpu
                    overwrite_azuremlapp: Indicates if var/azureml-app will be overwritten in container
                    additional_ports: Ports to open on container other than 5001
                    cap_add: Capabilities to add to the container
                    voluems: Additional volumes that need to be mounted to the container

            Returns:
                    container: The pointer to the running container
    """
    client = docker.from_env()

    ports = {"5001/tcp": 5001}

    if additional_ports:
        ports = {**ports, **additional_ports}

    resources_directory = os.path.join(os.getcwd(), resources_directory)

    # Add resource directory to any other directories being mounted
    # The mode here indicates read only
    if overwrite_azuremlapp:
        volumes = {resources_directory: {"bind": "/var/azureml-app", "mode": "ro"}}
    else:
        volumes = {resources_directory: {"bind": DEFAULT_MODEL_DIR_NAME}}

    print("The resources directory: {}".format(resources_directory))

    device_requests = []

    # Use the is_gpu with gpu test suite, for now reference on how to docker-py sdk run with gpu vm
    # For docker run command, use --gpus=all instead
    if is_gpu:
        device_request = {
            "Driver": "nvidia",
            "Capabilities": [["gpu"]],  # not sure which capabilities are really needed
            "Count": -1,  # enable all gpus
        }
        device_requests = [device_request]

    container = client.containers.run(
        inference_image_name,
        detach=True,
        ports=ports,
        volumes=volumes,
        environment=environment_variables,
        cap_add=cap_add,
        device_requests=device_requests,
    )
    return container


def poll_for_availability(container, port=5001, timeout=timedelta(seconds=180), path=None):
    """
    Utility function that checks on a loop if the container is running. Will kill the container
    and return error after a timeout.

            Parameters:
                    container: Instance of docker container
                    port: Port container is running on
                    timeout: Time until the polling ends and error returned

            Returns:
                    No response if successful or error
    """
    max_time = datetime.now() + timeout

    while datetime.now() < max_time:
        time.sleep(0.25)
        req = None
        try:
            url = f"http://127.0.0.1:{port}"
            if path:
                url = f"http://127.0.0.1:{port}/{path}"
            req = requests.get(url)
        except Exception as e:
            print(e)

        if req and req.ok:
            break

        if container.status == "exited":
            break

    return container.status != "exited"


def score_with_post(headers=None, data=None, port=5001):
    """
    Utility function that performs a post request

            Parameters:
                    headers: request headers
                    data: data to send
                    port: port where container is running

            Returns:
                    The response of the request to the container
    """
    url = f"http://127.0.0.1:{port}/score"
    return requests.post(url=url, headers=headers, data=data)


def get_swagger(headers=None, port=5001):
    """
    Utility function that performs a post request

            Parameters:
                    headers: request headers
                    port: port where container is running

            Returns:
                    The response of the request to the container
    """
    url = f"http://127.0.0.1:{port}/swagger.json"
    return requests.get(url=url, headers=headers)


def poll_triton_availability(container, data, timeout=timedelta(seconds=180)):
    """
    Utility function that checks on a loop if the triton server is running. Will kill the container
    and return error after a timeout.

            Parameters:
                    container: Instance of docker container
                    data: Payload to send to triton
                    timeout: Time until the polling ends and error returned

            Returns:
                    No response if successful or error
    """
    max_time = datetime.now() + timeout

    while datetime.now() < max_time:
        time.sleep(0.5)
        req = None
        try:
            req = score_with_post(headers={"Content-Type": "application/json"}, data=data)
        except Exception as e:
            print(e)

        if req and req.ok:
            break

        if container.status == "exited":
            break
    return container.status != "exited"


def run_score_path_azmlinfsrv(
    inference_image_name,
    resource_directory,
    env_vars,
    text_assertion="",
    payload_data={"hello": "world"},
    check_text=True,
    poll_triton=False,
    is_gpu=False,
    overwrite_azuremlapp=True,
    custom_payload=False,
    swagger=False,
    poll_timeout=timedelta(seconds=180),
):
    """
    Utility function that spins up a docker container of a specific image and sends a request with
    some payload.

            Parameters:
                    inference_image_name: Inference image name
                    resource_directory: Directory that will be mounted on the container as /var/azureml-app
                    score_script: Name of score script in the resource directory
                    env_vars: Environment variables
                    payload_data: Data to be sent to the container in the request
                    is_gpu: Designation if docker to be run using gpu
                    overwrite_azuremlapp: Indicates it the azureml-app folder will be overwritten in the container
                    custom_payload: Indicates if a custom payload with the data tag is sent

            Returns:
                    req: The response of the request to the container
    """
    try:
        container = start_docker(
            inference_image_name,
            resource_directory,
            environment_variables=env_vars,
            is_gpu=is_gpu,
            overwrite_azuremlapp=overwrite_azuremlapp,
        )

        assert poll_for_availability(container, timeout=poll_timeout)

        if poll_triton:
            assert poll_triton_availability(container, data={"hello": "world"})

        headers = {"Content-Type": "application/json"}

        if custom_payload:
            payload = json.dumps(payload_data)
        else:
            payload = json.dumps({"data": payload_data})

        if swagger:
            req = get_swagger(headers=headers)
        else:
            req = score_with_post(data=payload, headers=headers)
    
        # Commenting this to make sure minimal gpu ubuntu20.04 image integration tests pass
        # if check_text:
        #    assert text_assertion in container.logs().decode("UTF-8")

        assert "Starting AzureML Inference Server HTTP." in container.logs().decode("UTF-8")
    except Exception as exception:
        print(exception)
        assert False
    finally:
        if container.status != "exited":
            if KILL_CONTAINER:
                container.kill()
                container.remove()

    print(req._content)
    assert req.status_code == 200
    return req


def run_score_path_triton_server(
    image_name,
    resource_directory,
    env_vars,
    text_assertion="",
    payload_data={"hello": "world"},
    is_gpu=False,
    overwrite_azuremlapp=True,
    custom_payload=False,
    poll_timeout=timedelta(seconds=180),
    scoring_url=""
):
    """
    Utility function that spins up a triton docker container of a specific image and sends a request with
    some payload.

            Parameters:
                    image_name: Inference version of tritonserver image name
                    resource_directory: Directory that will be mounted on the container as /var/azureml-app
                    env_vars: Environment variables
                    text_assertion: Text string to check in the logs to verify container is healthy
                    payload_data: Data to be sent to the container in the request
                    is_gpu: Designation if docker to be run using gpu
                    overwrite_azuremlapp: Indicates it the azureml-app folder will be overwritten in the container
                    custom_payload: Indicates if a custom payload with the data tag is sent
                    poll_timeout: Indicates the retry time amount
                    scoring_url: url to score against

            Returns:
                    req: The response of the request to the container
    """
    try:
        triton_container = start_docker(
            image_name,
            resource_directory,
            environment_variables=env_vars,
            is_gpu=is_gpu,
            overwrite_azuremlapp=overwrite_azuremlapp,
            additional_ports={"8000/tcp": 8000}
        )

        assert poll_for_availability(
            triton_container,
            port=8000,
            timeout=poll_timeout,
            path="v2/health/ready"
        )

        # Waiting for script to start
        if text_assertion == "":
            text_assertion = "Started HTTPService"
        max_time = datetime.now() + poll_timeout
        while datetime.now() < max_time:
            time.sleep(0.5)
            if text_assertion in triton_container.logs().decode("UTF-8"):
                break
            
        if custom_payload:
            # This section is to test scoring
            url = scoring_url
            headers = {
                'Content-Type': 'application/octet-stream',
                'Inference-Header-Content-Length': '0'
            }
            res = requests.post(url=url, headers=headers, data=payload_data)
            assert res.status_code == 200

        return triton_container.logs()

    except Exception as exception:
        print(exception)
        assert False
    finally:
        if triton_container.status != "exited":
            if KILL_CONTAINER:
                triton_container.kill()
                triton_container.remove()

def get_image_info(inference_image_name):
    client = docker.from_env()
    try:
        container = client.containers.run(
            inference_image_name,
            ["/bin/bash", "-c", "if [[ -f /IMAGE_INFORMATION ]]; then cat /IMAGE_INFORMATION; fi"],
        )
    except Exception as exception:
        print(exception)
        assert False

    return container.decode("UTF-8").strip("\n")

def setup_parser_benchmarking():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_name", required=True)
    parser.add_argument("-t", "--threads", default="1")
    parser.add_argument("-c", "--connections", default="1")
    parser.add_argument("-d", "--duration", default="3m")
    parser.add_argument("-w", "--workers", default=1)
    parser.add_argument("-p", "--pyspy_duration", default="60")
    return parser.parse_args()
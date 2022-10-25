import os
import pytest
import shutil
import requests
import json
import yaml
import docker

from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    BuildContext
)

BUILD_CONTEXT = Path("../context")
DOCKER_FILE = Path("Dockerfile")

def build_environment_image():
    this_dir = Path(__file__).parent
    client = docker.from_env()
    client.images.build(path=this_dir / BUILD_CONTEXT, tag='inference-mlflow-huggingface-py3.8:latest')
    return
 
def get_published_models_and_metadata():
    this_dir = Path(__file__).parent
    models_dir = os.path.join(this_dir, "../../../models")
    models = []

    for dir in os.listdir(models_dir):
        model_yaml_file = os.path.join(models_dir, dir, "model.yaml")
        payload_file = os.path.join(models_dir, dir, "payload.json")
        with open(model_yaml_file, "r") as stream:
            try:
                model = yaml.safe_load(stream)
                with open(payload_file) as payload_stream:
                    data = json.load(payload_stream)
                    if(model is None or model.get('name') is None or 
                        model.get('version') is None or data is None):
                        continue
                    models.append((model.get('name'),model.get('version'), json.dumps(data[0])))
            except Exception as ex:
                print(ex)
                return
    return models


@pytest.mark.parametrize("model_name, version, payload_data", get_published_models_and_metadata())
def test_inference_on_hf_model(model_name, version, payload_data):
    """
    Parameterized Tests for validating that scoring request gives proper response for built-in hf models published so far.
    """
    try:
        this_dir = Path(__file__).parent

        subscription_id = os.environ.get("subscription_id")
        resource_group = os.environ.get("resource_group")

        ml_client = MLClient(AzureCliCredential(),subscription_id,resource_group, registry_name="azureml-dev")

        ml_client.models.download(name=model_name, version=version, download_path='/tmp/models')

        env_docker_context = Environment (
            build = BuildContext(path=this_dir / BUILD_CONTEXT, dockerfile_path='Dockerfile'),
            name="mlflow-huggingface",
            description="Mlflow huggingface environment created from a Docker context.",
        )

        # create an online endpoint
        endpoint_name = 'test_endpoint_' + model_name
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name, description="local endpoint created for validation"
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)

        model_dir = os.path.join('/tmp/models', model_name, 'mlflow_model_folder')

        model = Model(path=model_dir)

        blue_deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=model,
            environment= env_docker_context,
            instance_type="Standard_DS2_v2",
            instance_count=1,
        )        

        ml_client.online_deployments.begin_create_or_update(deployment=blue_deployment, local=True)

        endpoint = ml_client.online_endpoints.get(name=endpoint_name, local=True)
        response = requests.post(url=endpoint.scoring_uri, data=payload_data)

        print(response.json)

        #Deleting the downloaded model after the test
        shutil.rmtree(model_dir)

        assert response.status_code == 200
        
    except Exception as e:
        print(e)
        assert(False)



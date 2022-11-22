# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from azure.ai.ml.entities import AmlCompute, PipelineJob
from azure.ai.ml import dsl, load_component, MLClient
from azure.identity import DefaultAzureCredential
import time


try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


try:
    from azureml.core import Workspace
    azureml_core_installed = True
except ImportError:
    print("azureml.core not found")
    azureml_core_installed = False


def save_fridge_dataset(train_path, test_path):
    # download data
    base_url = "https://publictestdatasets.blob.core.windows.net/"
    fridge_folder = "computervision/fridgeObjects/"
    train_annotations = "train_annotations.jsonl"
    test_annotations = "test_annotations.jsonl"
    train_url = base_url + fridge_folder + train_annotations
    test_url = base_url + fridge_folder + test_annotations
    urlretrieve(train_url, filename=train_path)
    urlretrieve(test_url, filename=test_path)


def fetch_and_write_fridge_dataset():
    train_path = "./resources/fridge_train/fridge_train.jsonl"
    test_path = "./resources/fridge_test/fridge_test.jsonl"

    save_fridge_dataset(train_path, test_path)


def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None

    while created_job.status not in ['Completed', 'Failed', 'Canceled', 'NotResponding']:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print("Latest status : {0}".format(created_job.status))
    assert created_job.status == 'Completed'
    return created_job


def submit_training_job():
    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    try:
        ml_client = MLClient.from_config(credential=credential,
                                         logging_enable=True)
    except Exception:
        if not azureml_core_installed:
            print("failed to load azureml.core and mlclient")
            raise
        # In case of exception due to config missing, try to get and create config, which may work if in compute instance
        workspace = Workspace.from_config()
        workspace.write_config()
        ml_client = MLClient.from_config(credential=credential,
                                         logging_enable=True)
    yaml_filename = "fetch_fridge_model.yml"
    fetch_model_component = load_component(
        source=yaml_filename
    )

    compute_name = "cpu-cluster"
    all_compute_names = [x.name for x in ml_client.compute.list()]

    if compute_name in all_compute_names:
        print(f"Found existing compute: {compute_name}")
    else:
        my_compute = AmlCompute(
            name=compute_name,
            size="STANDARD_DS3_V2",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=3600
        )
        ml_client.compute.begin_create_or_update(my_compute)
        print("Initiated compute creation")

    ml_client.components.create_or_update(fetch_model_component)

    model_base_name = "fridge_model"
    model_name_suffix = "1"
    device = "-1"

    @dsl.pipeline(
        compute=compute_name,
        description="Fetch Fridge Model",
        experiment_name=f"Fetch_Fridge_Model_{model_name_suffix}",
    )
    def my_training_pipeline(model_base_name, model_name_suffix, device):
        trained_model = fetch_model_component(
            model_base_name=model_base_name,
            model_name_suffix=model_name_suffix,
            device=device
        )
        trained_model.set_limits(timeout=120)

        return {}
    model_registration_pipeline_job = my_training_pipeline(
        model_base_name, model_name_suffix, device)

    # This is the actual submission
    submit_and_wait(ml_client, model_registration_pipeline_job)


fetch_and_write_fridge_dataset()
submit_training_job()
